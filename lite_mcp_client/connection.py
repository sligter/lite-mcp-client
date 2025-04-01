"""服务器连接模块"""

import asyncio
import sys
import json
import os
import signal
import subprocess
from typing import Optional, Dict, List, Any, Tuple, Set
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class ServerConnection:
    """表示与单个MCP服务器的连接"""

    def __init__(self, config: Dict):
        """初始化服务器连接

        Args:
            config: 服务器配置字典
        """
        self.config = config
        self.session: Optional[ClientSession] = None
        # Initialize exit_stack here, but it will be replaced during successful connection
        self.exit_stack = AsyncExitStack()
        self.tools_cache = []
        self.resources_cache = []
        self.prompts_cache = []
        self.stdio = None
        self.write = None
        self.name = config.get('name', '未命名服务器')
        self.connected = False
        # Use anyio's Process type hint if possible, otherwise Any
        try:
            from anyio.abc import Process
            self.subprocess: Optional[Process] = None
        except ImportError:
            self.subprocess: Optional[Any] = None
        self.background_tasks: Set[asyncio.Task] = set()  # Track background tasks more reliably

    def _create_background_task(self, coro):
        """创建和跟踪后台任务"""
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    async def connect(self) -> bool:
        """连接到服务器

        Returns:
            连接是否成功
        """
        server_type = self.config.get('type', '').lower()

        try:
            if server_type == 'stdio':
                return await self._connect_stdio()
            elif server_type == 'sse':
                return await self._connect_sse()
            else:
                print(f"错误: 不支持的服务器类型: {server_type} for server '{self.name}'")
                return False
        except Exception as e:
            print(f"连接到服务器 '{self.name}' 时发生未处理的异常: {str(e)}")
            import traceback
            traceback.print_exc()
            # Ensure cleanup happens even on unexpected connection errors
            await self.cleanup()
            return False

    async def _connect_stdio(self) -> bool:
        """连接到STDIO类型的服务器

        Returns:
            连接是否成功
        """
        config = self.config
        command = config.get('command')
        if not command:
            print(f"错误: STDIO配置中缺少'command' for server '{self.name}'")
            return False
        args = config.get('args', [])
        env = config.get('env') or None
        cwd = config.get('cwd') or None
        print(f"正在连接到STDIO服务器: {command} {' '.join(args)}")
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            cwd=cwd
        )
        
        # 初始化临时变量
        temp_session = None
        temp_stdio = None
        temp_write = None
        temp_subprocess = None
        connection_succeeded = False
        stdio_transport = None
        attempt_exit_stack = None
        
        try:
            # 创建一个临时任务来执行连接
            async def do_connect():
                nonlocal temp_session, temp_stdio, temp_write, temp_subprocess, stdio_transport, attempt_exit_stack
                
                # 创建一个新的退出堆栈
                attempt_exit_stack = AsyncExitStack()
                
                # 进入stdio_client上下文
                
                stdio_transport = await attempt_exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                temp_stdio, temp_write = stdio_transport
                
                
                # 获取进程对象
                if hasattr(temp_stdio, '_process') and temp_stdio._process:
                    temp_subprocess = temp_stdio._process
                elif hasattr(stdio_transport, 'process'):
                    temp_subprocess = stdio_transport.process
                if temp_subprocess:
                    print(f"[{self.name}] 子进程已启动 (PID: {temp_subprocess.pid if temp_subprocess.pid else '未知'})。")
                
                # 进入ClientSession上下文
                
                temp_session = await attempt_exit_stack.enter_async_context(
                    ClientSession(temp_stdio, temp_write)
                )
                
                
                # 初始化会话

                await asyncio.wait_for(temp_session.initialize(), timeout=15.0)

                
                # 返回连接成功
                return True
            
            # 使用shield和单独的任务来执行连接
            connect_task = asyncio.create_task(do_connect())
            
            try:
                # 添加总体超时
                success = await asyncio.wait_for(
                    asyncio.shield(connect_task), 
                    timeout=20.0
                )
                if not connect_task.done():
                    # 等待任务正常完成
                    success = await connect_task
                
                if success:
                    connection_succeeded = True
            except asyncio.CancelledError:
                print(f"连接服务器 '{self.name}' 的操作被取消。")
                if not connect_task.done():
                    connect_task.cancel()
                    try:
                        await connect_task
                    except asyncio.CancelledError:
                        pass  # 预期的取消
                raise  # 重新抛出以便外层处理
            except asyncio.TimeoutError:
                print(f"连接或初始化服务器 '{self.name}' 超时。")
                if not connect_task.done():
                    connect_task.cancel()
                    try:
                        await connect_task
                    except (asyncio.CancelledError, Exception):
                        pass  # 忽略取消时的异常
            except Exception as e:
                print(f"连接到STDIO服务器 '{self.name}' 失败: {str(e)}")
                import traceback
                traceback.print_exc()
                if not connect_task.done():
                    connect_task.cancel()
                    try:
                        await connect_task
                    except (asyncio.CancelledError, Exception):
                        pass  # 忽略取消时的异常
            
            # 如果连接成功，转移资源并进行刷新操作
            if connection_succeeded:
                # 分配资源
                self.session = temp_session
                self.stdio = temp_stdio
                self.write = temp_write
                self.subprocess = temp_subprocess
                # 转移堆栈 - 这里是关键
                self.exit_stack = attempt_exit_stack
                self.connected = True
                
                # 刷新服务器信息
                print(f"[{self.name}] 正在刷新服务器信息...")
                try:
                    await asyncio.wait_for(self.refresh_server_info(), timeout=5.0)
                except asyncio.TimeoutError:
                    print(f"警告: 刷新服务器 '{self.name}' 信息超时。工具/资源可能不可用。")
                except Exception as refresh_err:
                    print(f"警告: 刷新服务器 '{self.name}' 信息时出错: {refresh_err}")
                
                print(f"成功连接到服务器: '{self.name}'")
                return True
        
        finally:
            # 清理失败的连接尝试
            if not connection_succeeded:
                print(f"[{self.name}] 清理失败的连接尝试...")
                
                # 终止子进程（如果存在）
                if temp_subprocess and hasattr(temp_subprocess, 'pid') and temp_subprocess.pid:
                    pid = temp_subprocess.pid
                    print(f"[{self.name}] 正在终止子进程 (PID: {pid})...")
                    try:
                        if sys.platform == 'win32':
                            subprocess.Popen(
                                ['taskkill', '/F', '/T', '/PID', str(pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW
                            )
                        else:
                            os.kill(pid, signal.SIGKILL)
                        print(f"[{self.name}] 终止信号已发送至 PID {pid}.")
                        await asyncio.sleep(0.1)
                    except ProcessLookupError:
                        print(f"[{self.name}] 子进程 (PID: {pid}) 在清理前已退出。")
                    except Exception as kill_err:
                        print(f"[{self.name}] 清理期间终止子进程 (PID: {pid}) 出错: {kill_err}")
                

                
                # 确保主对象状态被重置
                self.session = None
                self.stdio = None
                self.write = None
                self.subprocess = None
                self.connected = False
        
        # 如果由于错误/取消到达这里，返回False
        return False

    async def _connect_sse(self) -> bool:
        """连接到SSE类型的服务器

        Returns:
            连接是否成功
        """
        config = self.config
        url = config.get('url')
        if not url:
            print(f"错误: SSE配置中缺少'url' for server '{self.name}'")
            return False

        headers = config.get('headers', {})
        print(f"正在连接到SSE服务器: {url}")
        
        # 初始化临时变量
        temp_session = None
        connection_succeeded = False
        attempt_exit_stack = None
        
        try:
            # 检查是否有SSE客户端支持
            try:
                from mcp.client.sse import SSEServerParameters, sse_client
            except ImportError:
                print("错误: 缺少SSE客户端支持。请安装 'mcp[sse_client]' 或 'mcp[all]'.")
                return False
                
            # 创建一个临时任务来执行连接
            async def do_connect():
                nonlocal temp_session, attempt_exit_stack
                
                # 创建一个新的退出堆栈
                attempt_exit_stack = AsyncExitStack()
                
                server_params = SSEServerParameters(
                    url=url,
                    headers=headers
                )
                
                # 进入sse_client上下文
                
                sse_transport = await attempt_exit_stack.enter_async_context(sse_client(server_params))
                
                
                # 进入ClientSession上下文
                
                temp_session = await attempt_exit_stack.enter_async_context(ClientSession(sse_transport)) 
                
                
                # 初始化会话
                print(f"[{self.name}] 正在初始化SSE会话...")
                await asyncio.wait_for(temp_session.initialize(), timeout=10.0)
                print(f"[{self.name}] SSE会话已初始化。")
                
                # 返回连接成功
                return True
                
            # 使用shield和单独的任务来执行连接
            connect_task = asyncio.create_task(do_connect())
            
            try:
                # 添加总体超时
                success = await asyncio.wait_for(
                    asyncio.shield(connect_task), 
                    timeout=15.0
                )
                if not connect_task.done():
                    # 等待任务正常完成
                    success = await connect_task
                
                if success:
                    connection_succeeded = True
            except asyncio.CancelledError:
                print(f"连接SSE服务器 '{self.name}' 的操作被取消。")
                if not connect_task.done():
                    connect_task.cancel()
                    try:
                        await connect_task
                    except asyncio.CancelledError:
                        pass  # 预期的取消
                raise  # 重新抛出以便外层处理
            except asyncio.TimeoutError:
                print(f"连接或初始化SSE服务器 '{self.name}' 超时。")
                if not connect_task.done():
                    connect_task.cancel()
                    try:
                        await connect_task
                    except (asyncio.CancelledError, Exception):
                        pass  # 忽略取消时的异常
            except Exception as e:
                print(f"连接到SSE服务器 '{self.name}' 失败: {str(e)}")
                import traceback
                traceback.print_exc()
                if not connect_task.done():
                    connect_task.cancel()
                    try:
                        await connect_task
                    except (asyncio.CancelledError, Exception):
                        pass  # 忽略取消时的异常
            
            # 如果连接成功，转移资源并进行刷新操作
            if connection_succeeded:
                # 分配资源
                self.session = temp_session
                # 转移堆栈 - 这里是关键
                self.exit_stack = attempt_exit_stack
                self.connected = True
                
                # 刷新服务器信息
                print(f"[{self.name}] 正在刷新服务器信息...")
                try:
                    await asyncio.wait_for(self.refresh_server_info(), timeout=5.0)
                    print(f"[{self.name}] SSE服务器信息已刷新。")
                except asyncio.TimeoutError:
                    print(f"警告: 刷新SSE服务器 '{self.name}' 信息超时。")
                except Exception as refresh_err:
                    print(f"警告: 刷新SSE服务器 '{self.name}' 信息时出错: {refresh_err}")
                
                print(f"成功连接到SSE服务器: '{self.name}'")
                return True
        
        finally:
            # 清理失败的连接尝试
            if not connection_succeeded:
                
                # 确保主对象状态被重置
                self.session = None
                self.connected = False
        
        # 如果由于错误/取消到达这里，返回False
        return False

    async def refresh_server_info(self):
        """刷新服务器提供的工具、资源和提示信息"""
        if not self.session or not self.connected:
            # Don't print error if simply not connected yet
            # print(f"[{self.name}] Cannot refresh info, session not available or not connected.")
            return

        server_info = []
        fetch_errors = []

        # Get tools list
        try:
            response = await self.session.list_tools()
            self.tools_cache = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema # Assuming inputSchema is the standard field
                } for tool in response.tools
            ]
            server_info.append(f"{len(self.tools_cache)} 个工具")
        except Exception as e:
            # Check for "Method not found" specifically
            if "Method not found" in str(e):
                 print(f"[{self.name}] 服务器不支持 list_tools。")
                 self.tools_cache = []
                 server_info.append("工具列表不可用 (不支持)")
            else:
                err_msg = f"获取工具列表时出错: {str(e)}"
                print(f"[{self.name}] {err_msg}")
                fetch_errors.append(err_msg)
                self.tools_cache = []
                server_info.append("工具列表不可用 (错误)")


        # Get resources list
        try:
            response = await self.session.list_resources()
            self.resources_cache = [
                {
                    "uri": resource.uri,
                    "description": resource.description
                } for resource in response.resources
            ]
            server_info.append(f"{len(self.resources_cache)} 个资源")
        except Exception as e:
             if "Method not found" in str(e):
                 print(f"[{self.name}] 服务器不支持 list_resources。")
                 self.resources_cache = []
                 server_info.append("资源列表不可用 (不支持)")
             else:
                err_msg = f"获取资源列表时出错: {str(e)}"
                print(f"[{self.name}] {err_msg}")
                fetch_errors.append(err_msg)
                self.resources_cache = []
                server_info.append("资源列表不可用 (错误)")

        # Get prompts list
        try:
            response = await self.session.list_prompts()
            self.prompts_cache = []

            for prompt in response.prompts:
                prompt_info = {
                    "name": prompt.name,
                    "description": prompt.description,
                }

                # Safely get schema, preferring model_json_schema
                schema_value = None
                schema_source = None
                if hasattr(prompt, 'inputSchema'): # Standard field first
                    schema_source = prompt.inputSchema
                elif hasattr(prompt, 'input_schema'):
                     schema_source = prompt.input_schema
                elif hasattr(prompt, 'schema'): # Legacy/Pydantic v1
                    schema_source = prompt.schema

                if callable(schema_source):
                    try:
                        # Pydantic v2+ 优先使用 model_json_schema
                        if hasattr(schema_source, 'model_json_schema'):
                            schema_value = schema_source.model_json_schema()
                        elif hasattr(schema_source, '__call__'):
                            # 检查是否有 model_json_schema 函数属性
                            if hasattr(schema_source, 'model_json_schema'):
                                schema_value = schema_source.model_json_schema()
                            else:  # 回退到 Pydantic v1 或其他可调用对象
                                # 添加警告抑制
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                                    schema_value = schema_source()
                        else:
                            # 未知可调用类型
                            schema_value = {}
                    except Exception as schema_call_err:
                        print(f"[{self.name}] 调用提示 '{prompt.name}' 的 schema 方法失败: {schema_call_err}")
                        schema_value = {}  # 错误时默认为空字典
                elif isinstance(schema_source, dict):
                    schema_value = schema_source
                else:
                    # If schema_source is None or not a dict/callable
                    schema_value = {}

                prompt_info["schema"] = schema_value
                self.prompts_cache.append(prompt_info)

            server_info.append(f"{len(self.prompts_cache)} 个提示模板")
        except Exception as e:
            if "Method not found" in str(e):
                 print(f"[{self.name}] 服务器不支持 list_prompts。")
                 self.prompts_cache = []
                 server_info.append("提示模板不可用 (不支持)")
            else:
                err_msg = f"获取提示列表时出错: {str(e)}"
                print(f"[{self.name}] {err_msg}")
                fetch_errors.append(err_msg)
                self.prompts_cache = []
                server_info.append("提示模板不可用 (错误)")

        # Avoid printing the "发现..." message if connection failed before refresh
        if self.connected:
            print(f"服务器 '{self.name}' 信息: 发现 {', '.join(server_info)}")
            if fetch_errors:
                 print(f"[{self.name}] 刷新信息时遇到以下错误:\n - " + "\n - ".join(fetch_errors))


    async def call_tool(self, tool_name: str, params: Dict = None) -> Any:
        """调用指定的工具

        Args:
            tool_name: 工具名称
            params: 工具参数

        Returns:
            工具调用结果
        """
        if not self.session or not self.connected:
            print(f"警告: 服务器 '{self.name}' 未连接或会话不可用")
            return None

        if params is None:
            params = {}

        try:
            result = await self.session.call_tool(tool_name, params)
            
            # 处理TextContent类型
            if hasattr(result, 'content'):
                return result.content
            # 处理可能的其他MCP类型
            elif hasattr(result, 'to_dict'):
                return result.to_dict()
            # 如果已经是基本类型，直接返回
            return result
        except Exception as e:
            print(f"警告: 调用服务器 '{self.name}' 上的工具 '{tool_name}' 失败: {str(e)}")
            raise  # 重新抛出异常以便调用方处理


    async def get_resource(self, resource_uri: str) -> Any:
        """获取指定的资源

        Args:
            resource_uri: 资源URI

        Returns:
            资源内容
        """
        if not self.session or not self.connected:
            raise ValueError(f"服务器 '{self.name}' 未连接或会话不可用")

        try:
            result = await self.session.get_resource(resource_uri)
            return getattr(result, 'content', None) # Safely get content
        except Exception as e:
             raise ValueError(f"获取服务器 '{self.name}' 上的资源 '{resource_uri}' 失败: {str(e)}")

    async def call_prompt(self, prompt_name: str, params: Dict = None) -> str:
        """调用指定的提示模板

        Args:
            prompt_name: 提示模板名称
            params: 模板参数

        Returns:
            生成的提示文本
        """
        if not self.session or not self.connected:
            print(f"警告: 服务器 '{self.name}' 未连接或会话不可用")
            return ""

        if params is None:
            params = {}

        try:
            result = await self.session.get_prompt(prompt_name, params)
            # Safely get content, default to empty string
            return getattr(result, 'content', "")
        except Exception as e:
            print(f"警告: 调用服务器 '{self.name}' 上的提示模板 '{prompt_name}' 失败: {str(e)}")
            return ""  # 返回空字符串而不是抛出异常

    async def cleanup(self):
        """清理服务器连接和相关资源"""
        if not self.connected and not self.session and not self.subprocess:
            return True  # 没有需要清理的内容
        
        print(f"开始清理服务器 '{self.name}' 的连接...")
        initial_state = self.connected
        self.connected = False  # 立即标记为已断开
        
        # 1. 先取消关联的后台任务，避免它们干扰清理过程
        tasks_to_cancel = list(self.background_tasks)
        
        if tasks_to_cancel:
            print(f"[{self.name}] 正在取消 {len(tasks_to_cancel)} 个后台任务...")
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            try:
                # 设置较短的超时
                await asyncio.wait(tasks_to_cancel, timeout=0.5, return_when=asyncio.ALL_COMPLETED)
            except Exception:
                pass  # 忽略任务取消过程中的错误
        
        # 2. 更安全地终止子进程
        subprocess_to_terminate = self.subprocess
        self.subprocess = None  # 立即清除引用
        if subprocess_to_terminate and hasattr(subprocess_to_terminate, 'pid') and subprocess_to_terminate.pid:
            pid = subprocess_to_terminate.pid
            print(f"[{self.name}] 正在终止子进程 (PID: {pid})...")
            try:
                if sys.platform == 'win32':
                    subprocess.Popen(
                        ['taskkill', '/F', '/T', '/PID', str(pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                else:
                    os.kill(pid, signal.SIGKILL)
                # 不要等待进程终止，让操作系统处理
            except Exception as e:
                print(f"[{self.name}] 终止子进程时出错: {str(e)}")
        
        # 3. 关闭会话资源（避免等待）
        session_to_close = self.session
        self.session = None
        self.stdio = None
        self.write = None
        
        if session_to_close and hasattr(session_to_close, 'close') and callable(session_to_close.close):
            try:
                # 创建任务但不等待它完成
                asyncio.create_task(session_to_close.close())
            except Exception:
                pass  # 忽略关闭错误
        
        # 4. 重置exit_stack而不是尝试aclose()
        self.exit_stack = AsyncExitStack()
        
        # 额外清理以确保资源被正确释放
        self.tools_cache = []
        self.resources_cache = []
        self.prompts_cache = []
        
        return True 