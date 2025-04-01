import asyncio
import sys
import json
import os
from typing import Optional, Dict, List, Any, Tuple, Set
from contextlib import AsyncExitStack
import signal # For Unix process termination

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, terminate_windows_process # Import windows specific termination
from mcp.client.sse import sse_client

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import re
import argparse
from pathlib import Path
import subprocess # For taskkill on Windows

load_dotenv()  # 加载.env文件中的环境变量

# 默认配置文件路径
DEFAULT_CONFIG_PATH = "mcp_config.json"

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
            raise ValueError(f"服务器 '{self.name}' 未连接或会话不可用")

        if params is None:
            params = {}

        try:
            result = await self.session.call_tool(tool_name, params)
            # Access content safely
            content = getattr(result, 'content', None)
            # Basic check for common attribute patterns if content is complex
            if hasattr(content, 'content'):
                return content.content
            elif hasattr(content, 'text'):
                return content.text
            # Return content directly if simple, or its string representation if complex but unhandled
            return content if not hasattr(content, '__dict__') else str(content)

        except Exception as e:
            # Improve error message
            raise ValueError(f"调用服务器 '{self.name}' 上的工具 '{tool_name}' 失败: {str(e)}")


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
            # 使用 get_prompt 而不是 call_prompt
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
        
        # 1. 终止子进程（强制且不等待太长时间）
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
                    print(f"[{self.name}] taskkill 命令已发送至 PID {pid}.")
                else:
                    os.kill(pid, signal.SIGKILL)
                print(f"[{self.name}] SIGKILL 发送至 PID {pid}.")
                await asyncio.sleep(0.1)
            except ProcessLookupError:
                print(f"[{self.name}] 子进程 (PID: {pid}) 已不存在。")
            except Exception as e:
                print(f"[{self.name}] 终止子进程 (PID: {pid}) 时出错: {str(e)}")
        
        # 2. 取消关联的后台任务
        tasks_to_cancel = list(self.background_tasks)
        self.background_tasks.clear()
        if tasks_to_cancel:
            print(f"[{self.name}] 正在取消 {len(tasks_to_cancel)} 个后台任务...")
            for task in tasks_to_cancel:
                if not task.done():
                    task.cancel()
            try:
                done, pending = await asyncio.wait(tasks_to_cancel, timeout=0.5, return_when=asyncio.ALL_COMPLETED)
                if pending:
                    print(f"[{self.name}] 警告: {len(pending)} 个后台任务未能及时取消。")
            except (asyncio.CancelledError, Exception) as e:
                print(f"[{self.name}] 等待后台任务取消时出错: {e}")
        
        # 3. 关闭会话（带超时）
        session_to_close = self.session
        self.session = None
        self.stdio = None
        self.write = None
        if session_to_close and hasattr(session_to_close, 'close') and callable(session_to_close.close):
            print(f"[{self.name}] 正在关闭会话...")
            try:
                # 创建单独的任务来关闭会话
                close_task = asyncio.create_task(session_to_close.close())
                await asyncio.wait_for(close_task, timeout=1.0)
                print(f"[{self.name}] 会话已关闭。")
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                print(f"[{self.name}] 关闭会话时出错: {e}")
        
        # 4. 完全放弃旧的AsyncExitStack - 关键修复
        # 不要尝试await exit_stack.aclose()！
        self.exit_stack = AsyncExitStack()
        
        final_status = "已断开" if initial_state else "清理完成 (本已断开)"

        return True


class GenericMCPClient:
    """通用型MCP客户端，可以连接到任意MCP服务器"""

    def __init__(self):
        """初始化lite_mcp_client"""
        self.connections: Dict[str, ServerConnection] = {}  # 存储多个连接，键为服务器名称
        self.current_connection: Optional[ServerConnection] = None  # 当前活动连接
        self.server_config: Optional[Dict] = None # Store config of current connection

        # --- LLM Initialization ---
        self.llm = None
        try:
            provider = os.environ.get("PROVIDER", "openai").lower()
            api_key = os.environ.get("OPENAI_API_KEY")
            api_base = os.environ.get("OPENAI_API_BASE") # Optional
            model_name = os.environ.get("MODEL_NAME") # Optional, sensible defaults below

            google_api_key = os.environ.get("GOOGLE_API_KEY")
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
            anthropic_api_base = os.environ.get("ANTHROPIC_API_BASE") # Optional
            aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            aws_session_token = os.environ.get("AWS_SESSION_TOKEN") # Optional
            aws_region_name = os.environ.get("AWS_REGION_NAME", "us-east-1") # Default region


            print(f"使用的LLM提供商: {provider}")

            if "openai" in provider and api_key:
                model_name = model_name or "gpt-3.5-turbo"
                print(f"初始化 OpenAI: model={model_name}, base_url={api_base or '默认'}")
                self.llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=api_base,
                    # max_tokens=16000, # Often model-dependent, manage in prompt if needed
                    temperature=0.7,
                    request_timeout=120, # Longer timeout
                    max_retries=2,
                )
            elif "google" in provider and google_api_key:
                model_name = model_name or "gemini-pro" # Example default
                print(f"初始化 Google GenAI: model={model_name}")
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=google_api_key,
                    # max_output_tokens=8192, # Manage in prompt if needed
                    temperature=0.7,
                    # request_timeout=120, # Not directly settable like this? Check docs
                    max_retries=2,
                    # Settings for safety/etc. can be added here
                )
            elif "anthropic" in provider and anthropic_api_key:
                model_name = model_name or "claude-3-sonnet-20240229" # Example default
                print(f"初始化 Anthropic: model={model_name}, base_url={anthropic_api_base or '默认'}")
                self.llm = ChatAnthropic(
                    model=model_name,
                    anthropic_api_key=anthropic_api_key,
                    anthropic_api_url=anthropic_api_base,
                    # max_tokens_to_sample=4096, # Manage in prompt
                    temperature=0.7,
                    request_timeout=120,
                    max_retries=2,
                )
            elif "aws" in provider and aws_access_key_id and aws_secret_access_key:
                 model_name = model_name or "anthropic.claude-3-sonnet-20240229-v1:0" # Example Bedrock model ID
                 print(f"初始化 AWS Bedrock Converse: model={model_name}, region={aws_region_name}")
                 self.llm = ChatBedrockConverse(
                    model=model_name,
                    region_name=aws_region_name,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token, # Pass if present
                    # max_tokens=4096, # Manage in prompt
                    temperature=0.7,
                    # timeout=120, # Check specific param name in docs
                    max_retries=2,
                )

            if not self.llm:
                 print("\n警告: 未找到有效的LLM API密钥或配置，LLM回答 ('ask') 功能将不可用。")
                 print("请在 .env 文件中设置相应的环境变量 (例如 OPENAI_API_KEY, GOOGLE_API_KEY 等)。")

        except ImportError as import_err:
             print(f"\n警告: 缺少必要的 LangChain 库 ({import_err})。LLM回答 ('ask') 功能可能不可用。")
             print("请确保已安装对应提供商的库 (e.g., 'pip install langchain-openai langchain-google-genai')")
             self.llm = None
        except Exception as llm_init_err:
             print(f"\n警告: 初始化LLM时出错: {llm_init_err}")
             self.llm = None
        # --- End LLM Initialization ---


        # Add conversation history
        self.history: List[Any] = [] # Use Any for LangChain messages

    def load_server_config(self, config_path: str = DEFAULT_CONFIG_PATH) -> Dict:
        """从配置文件加载服务器配置

        Args:
            config_path: 配置文件路径

        Returns:
            配置的服务器字典
        """
        # If config file doesn't exist, create a default one
        config_file = Path(config_path)
        if not config_file.is_file():
            print(f"配置文件 '{config_path}' 不存在，将创建默认配置。")
            self._create_default_config(config_file)

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            if 'mcp_servers' not in config or not isinstance(config['mcp_servers'], list):
                print(f"警告: 配置文件 '{config_path}' 缺少有效的 'mcp_servers' 列表。")
                config['mcp_servers'] = [] # Ensure it exists as an empty list

            return config
        except json.JSONDecodeError:
            raise ValueError(f"配置文件 '{config_path}' 不是有效的JSON格式。")
        except Exception as e:
            raise ValueError(f"加载配置文件 '{config_path}' 时出错: {str(e)}")

    def _create_default_config(self, config_file: Path):
        """创建默认配置文件

        Args:
            config_file: Path object for the config file
        """
        # Get the directory containing the client script
        script_dir = Path(__file__).parent.resolve()
        # Example server script path (adjust if your server is elsewhere)
        default_server_script = script_dir / "server.py" # Assumes server.py is in the same dir

        default_config = {
            "mcp_servers": [
                {
                    "name": "默认STDIO服务",
                    "type": "stdio",
                    # Use sys.executable for portability
                    "command": sys.executable, # Use the current python interpreter
                    "args": [str(default_server_script)], # Convert Path to string
                    "env": None, # Explicitly None if no extra env vars needed
                    "cwd": str(script_dir), # Run server in the script's directory
                    "description": "一个通过 STDIO 运行的示例 MCP 服务"
                },
                # Add more examples if needed, e.g., for SSE
                # {
                #     "name": "示例SSE服务",
                #     "type": "sse",
                #     "url": "http://localhost:8000/events",
                #     "headers": {"Authorization": "Bearer your_token_here"},
                #     "description": "一个通过 SSE 连接的示例 MCP 服务"
                # }
            ],
            "default_server": ["默认STDIO服务"] # Make it a list by default
        }

        try:
            config_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print(f"已创建默认配置文件: {config_file}")
        except Exception as e:
             print(f"错误: 无法创建默认配置文件 '{config_file}': {e}")


    def list_available_servers(self, config: Dict) -> None:
        """列出配置中的所有可用服务器

        Args:
            config: 配置字典
        """
        servers = config.get('mcp_servers', [])
        if not servers:
            print("配置文件中没有定义服务器。")
            return

        print("\n配置文件中可用的服务器:")
        for i, server in enumerate(servers, 1):
            name = server.get('name', f'未命名服务器 {i}')
            server_type = server.get('type', '未知类型').lower()
            desc = server.get('description', '无描述')

            print(f"{i}. {name} [{server_type}] - {desc}")
            if server_type == 'stdio':
                command = server.get('command', 'N/A')
                args = ' '.join(server.get('args', []))
                cwd = server.get('cwd', '默认')
                print(f"   命令: {command} {args}")
                print(f"   工作目录: {cwd}")
            elif server_type == 'sse':
                url = server.get('url', 'N/A')
                print(f"   URL: {url}")
            # Add other types if needed

        # Show default servers
        default = config.get('default_server', [])
        if isinstance(default, str): # Handle non-list default_server
             default = [default]

        if default:
            print(f"\n默认连接服务器: {', '.join(default)}")
        else:
             print("\n没有配置默认连接的服务器。")


    async def connect_to_server_by_name(self, server_name: str, config: Dict) -> bool:
        """通过名称连接到服务器

        Args:
            server_name: 服务器名称
            config: 配置字典

        Returns:
            连接是否成功
        """
        # Find server config by name
        server_config = None
        for srv in config.get('mcp_servers', []):
            if srv.get('name') == server_name:
                server_config = srv
                break

        if not server_config:
            print(f"错误: 在配置文件中找不到服务器 '{server_name}' 的配置。")
            self.list_available_servers(config) # Show available servers for help
            return False

        # Check if already connected and active
        if server_name in self.connections and self.connections[server_name].connected:
            print(f"已经连接到服务器 '{server_name}'。")
            # Ensure it's the current connection
            if self.current_connection is not self.connections[server_name]:
                 print("正在切换到此服务器...")
                 self.current_connection = self.connections[server_name]
                 self.server_config = self.connections[server_name].config
            return True

        # If connection exists but is disconnected, remove the old object *without* cleanup call here.
        # Let the new connect attempt handle resource creation cleanly.
        if server_name in self.connections:
            print(f"找到服务器 '{server_name}' 的旧连接记录，将移除并尝试重新连接...")
            # Just remove the reference, don't trigger cleanup from here
            del self.connections[server_name]
            # If it was the current connection, reset current_connection
            if self.current_connection and self.current_connection.name == server_name:
                 self.current_connection = None
                 self.server_config = None


        # Proceed with connecting using the found configuration
        print(f"正在连接到服务器 '{server_name}'...")
        return await self.connect_to_server_config(server_config)


    async def connect_to_server_config(self, server_config: Dict) -> bool:
        """根据配置连接到服务器

        Args:
            server_config: 服务器配置字典

        Returns:
            连接是否成功
        """
        server_name = server_config.get('name')
        if not server_name:
            print("错误: 服务器配置缺少 'name' 字段。")
            return False

        # Double-check if already connected (safety check)
        if server_name in self.connections and self.connections[server_name].connected:
            print(f"重复连接请求：已经连接到 '{server_name}'。")
            self.current_connection = self.connections[server_name] # Ensure it's current
            self.server_config = self.connections[server_name].config
            return True

        # Remove any potentially disconnected old entry
        if server_name in self.connections:
             print(f"[{server_name}] 移除旧的非活动连接记录...")
             del self.connections[server_name]
             if self.current_connection and self.current_connection.name == server_name:
                  self.current_connection = None
                  self.server_config = None


        # Create and connect
        connection = None # Initialize connection to None
        try:
            connection = ServerConnection(server_config)
            connect_success = await connection.connect()

            if connect_success:
                self.connections[server_name] = connection
                # Set as current connection only if no other connection is active,
                # or make it controllable via a flag/logic later.
                # For simplicity now, the last successful connection becomes current.
                self.current_connection = connection
                self.server_config = connection.config
                return True
            else:
                # connect() method should handle its own cleanup on failure
                print(f"无法连接到服务器 '{server_name}' (连接尝试失败)。")
                # Ensure the failed connection object isn't left in the dictionary
                if server_name in self.connections:
                     del self.connections[server_name]
                return False
        except Exception as e:
            print(f"连接到服务器 '{server_name}' 时发生意外异常: {str(e)}")
            import traceback
            traceback.print_exc()
            # Ensure partial connection resources are cleaned up if an exception occurred *here*
            if connection:
                 print(f"[{server_name}] 清理部分连接资源...")
                 await connection.cleanup()
            # Remove from dict if it was somehow added before the exception
            if server_name in self.connections:
                del self.connections[server_name]
            return False

    async def disconnect_server(self, server_name: str) -> bool:
        """断开与指定服务器的连接

        Args:
            server_name: 服务器名称

        Returns:
            是否成功断开连接 (True even if cleanup had minor issues)
        """
        if server_name not in self.connections:
            print(f"服务器 '{server_name}' 未连接或已被移除。")
            return True # Consider it "disconnected" if not in the list

        connection_to_disconnect = self.connections[server_name]

        print(f"准备断开服务器 '{server_name}'...")

        # 1. Remove from active connections dictionary *first*
        del self.connections[server_name]

        # 2. Reset current connection if it was the one being disconnected
        if self.current_connection is connection_to_disconnect:
            self.current_connection = None
            self.server_config = None
            print(f"[{server_name}] 已取消设为当前活动连接。")
             # Optionally, switch to another connection if available
            if self.connections:
                next_server_name = next(iter(self.connections))
                self.current_connection = self.connections[next_server_name]
                self.server_config = self.current_connection.config
                print(f"已自动切换到活动服务器: '{next_server_name}'")


        # 3. Perform the cleanup with timeout and error handling
        try:
            cleanup_success = await asyncio.wait_for(connection_to_disconnect.cleanup(), timeout=5.0) # Adjust timeout as needed
            if cleanup_success:
                 print(f"服务器 '{server_name}' 已成功断开并清理。")
            else:
                 print(f"服务器 '{server_name}' 断开，但清理过程中可能遇到问题。")
            # Short pause to allow system resources (sockets, processes) to fully release
            await asyncio.sleep(0.2)
            return True # Return True as the connection is removed from the client's perspective
        except asyncio.TimeoutError:
            print(f"警告: 断开服务器 '{server_name}' 连接时清理超时。资源可能未完全释放。")
            return True # Still return True, as it's removed from active list
        except asyncio.CancelledError:
            print(f"警告: 断开服务器 '{server_name}' 操作被取消。")
            # Manually ensure the connection object state is marked disconnected
            connection_to_disconnect.connected = False
            return False # Indicate cancellation happened
        except Exception as e:
            print(f"错误: 断开服务器 '{server_name}' 连接并清理时发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return True # Return True as it's removed from active list

    async def switch_connection(self, server_name: str) -> bool:
        """切换到另一个已连接的服务器

        Args:
            server_name: 要切换到的服务器名称

        Returns:
            切换是否成功
        """
        if server_name not in self.connections:
            print(f"错误: 未连接到服务器 '{server_name}'。无法切换。")
            self.list_connections()
            return False

        connection = self.connections[server_name]
        if not connection.connected:
            # Attempt to reconnect if disconnected? Or just fail? For now, fail.
            print(f"错误: 服务器 '{server_name}' 当前未连接。请先使用 'connect {server_name}'。")
            return False

        if self.current_connection is connection:
             print(f"已经是当前活动服务器: '{server_name}'")
             return True

        self.current_connection = connection
        self.server_config = connection.config # Store its config
        print(f"已切换到活动服务器: '{server_name}'")
        return True

    async def refresh_server_info(self):
        """刷新当前服务器提供的工具、资源和提示信息"""
        if not self.current_connection:
            print("错误: 没有活动的服务器连接。请先使用 'connect' 或 'switch'。")
            return

        if not self.current_connection.connected:
            print(f"错误: 当前服务器 '{self.current_connection.name}' 未连接。")
            return

        print(f"正在刷新服务器 '{self.current_connection.name}' 的信息...")
        await self.current_connection.refresh_server_info()


    async def call_tool(self, tool_name: str, params: Dict = None) -> Any:
        """调用当前连接服务器上的指定工具 (Helper for interactive mode)"""
        if not self.current_connection:
            raise ValueError("没有活动的服务器连接。")
        if not self.current_connection.connected:
             raise ValueError(f"当前服务器 '{self.current_connection.name}' 未连接。")

        return await self.current_connection.call_tool(tool_name, params)

    async def get_resource(self, resource_uri: str) -> Any:
        """获取当前连接服务器上的指定资源 (Helper for interactive mode)"""
        if not self.current_connection:
            raise ValueError("没有活动的服务器连接。")
        if not self.current_connection.connected:
             raise ValueError(f"当前服务器 '{self.current_connection.name}' 未连接。")

        return await self.current_connection.get_resource(resource_uri)

    async def call_prompt(self, prompt_name: str, params: Dict = None) -> str:
        """调用当前连接服务器上的指定提示模板 (Helper for interactive mode)"""
        if not self.current_connection:
            raise ValueError("没有活动的服务器连接。")
        if not self.current_connection.connected:
             raise ValueError(f"当前服务器 '{self.current_connection.name}' 未连接。")

        return await self.current_connection.call_prompt(prompt_name, params)

    async def process_with_llm(self, prompt: str, data: Any) -> str:
        """使用LLM处理数据 (Helper for potential future use)"""
        if not self.llm:
            return "错误: LLM 未初始化。无法处理数据。"

        try:
            # Convert data to string representation for the LLM
            if isinstance(data, (dict, list)):
                try:
                    data_str = json.dumps(data, ensure_ascii=False, indent=2, default=str) # Use default=str for non-serializable
                except TypeError:
                    data_str = str(data) # Fallback to simple string conversion
            else:
                data_str = str(data)

            # Simple LLM invocation
            messages = [HumanMessage(content=f"{prompt}\n\n以下是相关数据:\n```\n{data_str}\n```")]
            response = await self.llm.ainvoke(messages)
            return response.content # Assuming response has a 'content' attribute
        except Exception as e:
            print(f"调用LLM处理数据时发生错误: {e}")
            return f"错误: 调用LLM失败 ({e})"


    async def smart_query(self, query_text: str, max_attempts: int = 30) -> str:
        """LLM回答处理器 - 调用LLM解释回答并调用适当的工具"""
        if not self.connections:
            print("提示: 没有已连接的服务器,使用纯模型回答")

        print(f"\n正在处理LLM回答: {query_text}\n")

        # 构建上下文
        available_tools = self.collect_all_tools()
        available_prompts = self.collect_all_prompts()
        available_resources = self.collect_all_resources()

        # 使用类的历史记录而不是创建临时变量
        # 添加用户回答到历史记录
        self.history.append({"role": "user", "content": query_text})
        
        # 尝试循环
        attempt = 0
        final_response = None
        
        # 在调用LLM前添加调试信息
        print(f"可用工具数量: {sum(len(tools) for tools in self.collect_all_tools().values())}")
        print(f"可用提示模板数量: {sum(len(prompts) for prompts in self.collect_all_prompts().values())}")
        
        while attempt < max_attempts:
            attempt += 1
            print(f"[第 {attempt}/{max_attempts} 步]")
            
            # 调用LLM获取下一步
            print("正在请求 LLM 进行下一步操作...")
            try:
                llm_response = await self._call_llm_for_next_step(
                    self.history, available_tools, available_prompts, available_resources
                )
                print('llm_response:',llm_response)
                print("\nLLM 响应类型:", llm_response.get("action", "unknown"))
            except Exception as e:
                import traceback
                print(f"调用LLM时出错: {str(e)}")
                traceback.print_exc()
                return f"LLM回答失败: {str(e)}"
            
            action = llm_response.get("action", "unknown")
            
            # 处理错误情况
            if action == "error":
                error_msg = llm_response.get("error", "未知错误")
                print(f"处理LLM响应时出错: {error_msg}")
                if attempt < max_attempts:
                    print("尝试继续回答...")
                    self.history.append({"role": "assistant", "content": llm_response.get("raw_response", "")})
                    self.history.append({"role": "user", "content": f"出现错误: {error_msg}。请修正格式后重试。"})
                    continue
                else:
                    return f"LLM回答失败: 处理LLM响应时出错: {error_msg}"
            
            # 处理最终答案
            elif action == "final_answer":
                final_content = llm_response.get("content", "")
                print(f"\n[最终答案]:\n{final_content}")
                # 添加最终答案到历史记录
                self.history.append({"role": "assistant", "content": final_content})
                return final_content
            
            # 处理工具调用
            elif action == "call_tool":
                server_name = llm_response.get("server")
                tool_name = llm_response.get("tool")
                params = llm_response.get("params", {})
                
                print(f"\n正在调用工具: {server_name}.{tool_name}")
                print(f"参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
                
                try:
                    result = await self.call_tool_by_server(server_name, tool_name, params)
                    
                    # 格式化结果
                    try:
                        result_str = json.dumps(result, ensure_ascii=False, indent=2, default=str)
                    except:
                        result_str = str(result)
                                            
                    # 添加结果到历史记录
                    print(f"工具 {server_name}.{tool_name} 调用成功。结果如下:\n```json\n{result_str}\n```")
                    result_msg = f"工具 {server_name}.{tool_name} 调用成功。结果如下:\n```json\n{result_str}\n```"
                    self.history.append({"role": "assistant", "content": llm_response.get("raw_response", "")})
                    self.history.append({"role": "user", "content": result_msg})
                    
                except Exception as e:
                    error_msg = f"调用工具 {server_name}.{tool_name} 失败: {str(e)}"
                    print(f"\n{error_msg}")
                    # 更友好的方式处理JSON参数错误
                    if "JSON" in str(e) or "json" in str(e):
                        print("尝试修复JSON格式问题...")
                        try:
                            # 尝试清理JSON参数
                            params_str = json.dumps(params)
                            params_str = re.sub(r',\s*}', '}', params_str)
                            params_str = re.sub(r',\s*]', ']', params_str)
                            fixed_params = json.loads(params_str)
                            print("使用修复后的参数重试...")
                            result = await self.call_tool_by_server(server_name, tool_name, fixed_params)
                            
                            # 成功修复后处理结果
                            try:
                                result_str = json.dumps(result, ensure_ascii=False, indent=2, default=str)
                            except:
                                result_str = str(result)
                            
                            # 添加修复后的结果到历史记录
                            result_msg = f"工具 {server_name}.{tool_name} 调用成功(参数已修复)。结果如下:\n```json\n{result_str}\n```"
                            self.history.append({"role": "assistant", "content": llm_response.get("raw_response", "")})
                            self.history.append({"role": "user", "content": result_msg})
                            continue  # 继续下一轮对话
                        except Exception as retry_err:
                            print(f"重试失败: {retry_err}")
                            self.history.append({"role": "assistant", "content": llm_response.get("raw_response", "")})
                            self.history.append({"role": "user", "content": f"错误: {error_msg}。请修正参数格式并重试。"})
                    else:
                        self.history.append({"role": "assistant", "content": llm_response.get("raw_response", "")})
                        self.history.append({"role": "user", "content": f"错误: {error_msg}"})
            
            # 处理提示调用
            elif action == "call_prompt":
                server_name = llm_response.get("server")
                prompt_name = llm_response.get("prompt")
                params = llm_response.get("params", {})
                
                print(f"\n正在调用提示模板: {server_name}.{prompt_name}")
                print(f"参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
                
                try:
                    result = await self.call_prompt_by_server(server_name, prompt_name, params)
                                        
                    # 添加结果到历史记录
                    result_msg = f"提示模板 {server_name}.{prompt_name} 调用成功。结果如下:\n```\n{result}\n```"
                    self.history.append({"role": "assistant", "content": llm_response.get("raw_response", "")})
                    self.history.append({"role": "user", "content": result_msg})
                    
                except Exception as e:
                    error_msg = f"调用提示模板 {server_name}.{prompt_name} 失败: {str(e)}"
                    print(f"\n{error_msg}")
                    self.history.append({"role": "assistant", "content": llm_response.get("raw_response", "")})
                    self.history.append({"role": "user", "content": f"错误: {error_msg}"})
            
            # 处理未知操作
            else:
                print(f"未知的操作类型: {action}")
                self.history.append({"role": "assistant", "content": llm_response.get("raw_response", "")})
                self.history.append({"role": "user", "content": f"未知的操作类型: {action}。请使用正确的格式。"})
        
        # 如果达到最大尝试次数，请求最终答案
        print(f"\n已达到最大尝试次数 ({max_attempts})。请求最终答案...")
        self.history.append({"role": "user", "content": "已达到最大尝试次数。请根据已有信息提供最终答案。"})
        
        final_llm_response = await self._call_llm_for_next_step(
            self.history, available_tools, available_prompts, available_resources
        )
        
        if final_llm_response.get("action") == "final_answer":
            final_content = final_llm_response.get("content", "无法获取最终答案")
            # 添加最终答案到历史记录
            self.history.append({"role": "assistant", "content": final_content})
            return final_content
        else:
            # 如果仍然不是最终答案，返回原始响应
            raw_response = final_llm_response.get("raw_response", "达到最大尝试次数，但无法获取最终答案")
            self.history.append({"role": "assistant", "content": raw_response})
            return raw_response


    def collect_all_tools(self) -> Dict[str, List[Dict]]:
        """收集所有已连接服务器的可用工具
        
        Returns:
            Dict[服务器名, 工具列表]
        """
        all_tools = {}
        for server_name, connection in self.connections.items():
            if connection.connected and connection.tools_cache:
                all_tools[server_name] = connection.tools_cache
        return all_tools

    def collect_all_prompts(self) -> Dict[str, List[Dict]]:
        """收集所有已连接服务器的可用提示模板
        
        Returns:
            Dict[服务器名, 提示模板列表]
        """
        all_prompts = {}
        for server_name, connection in self.connections.items():
            if connection.connected and connection.prompts_cache:
                all_prompts[server_name] = connection.prompts_cache
        return all_prompts

    def collect_all_resources(self) -> Dict[str, List[Dict]]:
        """收集所有已连接服务器的可用资源
        
        Returns:
            Dict[服务器名, 资源列表]
        """
        all_resources = {}
        for server_name, connection in self.connections.items():
            if connection.connected and connection.resources_cache:
                all_resources[server_name] = connection.resources_cache
        return all_resources

    def _format_tools_for_llm(self) -> str:
        """格式化工具信息，供LLM使用
        
        Returns:
            格式化的工具信息字符串
        """
        all_tools = self.collect_all_tools()
        if not all_tools:
            return "没有可用的工具。"
        
        formatted_lines = []
        for server_name, tools in all_tools.items():
            formatted_lines.append(f"\n## 服务器: {server_name}")
            if not tools:
                formatted_lines.append("无可用工具。")
                continue
            
            for tool in tools:
                tool_name = tool.get('name', '未命名工具')
                tool_desc = tool.get('description', '无描述')
                formatted_lines.append(f"- **{server_name}.{tool_name}**")
                formatted_lines.append(f"  描述: {tool_desc}")
                
                # 处理参数
                params = tool.get('parameters', {})
                if params:
                    formatted_lines.append(f"  参数:")
                    for param_name, param_info in params.items():
                        is_required = param_info.get('required', False)
                        param_desc = param_info.get('description', '无描述')
                        param_mark = "*" if is_required else ""
                        formatted_lines.append(f"  - {param_name}{param_mark}: {param_desc}")
                
                formatted_lines.append("")  # 空行分隔
        
        return "\n".join(formatted_lines)

    def _format_prompts_for_llm(self) -> str:
        """格式化提示模板信息，供LLM使用
        
        Returns:
            格式化的提示模板信息字符串
        """
        all_prompts = self.collect_all_prompts()
        if not all_prompts:
            return "没有可用的提示模板。"
        
        formatted_lines = []
        for server_name, prompts in all_prompts.items():
            formatted_lines.append(f"\n## 服务器: {server_name}")
            if not prompts:
                formatted_lines.append("无可用提示模板。")
                continue
            
            for prompt in prompts:
                prompt_name = prompt.get('name', '未命名提示')
                prompt_desc = prompt.get('description', '无描述')
                formatted_lines.append(f"- **{server_name}.{prompt_name}**")
                formatted_lines.append(f"  描述: {prompt_desc}")
                
                # 处理参数
                schema = prompt.get('schema', {})
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                if properties:
                    formatted_lines.append(f"  参数:")
                    for param_name, param_info in properties.items():
                        is_required = param_name in required
                        param_desc = param_info.get('description', '无描述')
                        param_mark = "*" if is_required else ""
                        formatted_lines.append(f"  - {param_name}{param_mark}: {param_desc}")
                
                formatted_lines.append("")  # 空行分隔
        
        return "\n".join(formatted_lines)


    async def call_tool_by_server(self, server_name: str, tool_name: str, params: Dict = None) -> Any:
        """调用指定服务器上的工具

        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            params: 工具参数

        Returns:
            工具调用结果
        Raises:
            ValueError: If server not connected or tool call fails.
        """
        if server_name not in self.connections:
            raise ValueError(f"未连接到服务器 '{server_name}'。可用连接: {list(self.connections.keys())}")

        connection = self.connections[server_name]
        if not connection.connected:
            raise ValueError(f"服务器 '{server_name}' 连接已断开。")

        # The actual call logic is in ServerConnection.call_tool
        return await connection.call_tool(tool_name, params)


    async def call_prompt_by_server(self, server_name: str, prompt_name: str, params: Dict = None) -> str:
        """调用指定服务器上的提示模板

        Args:
            server_name: 服务器名称
            prompt_name: 提示模板名称
            params: 模板参数

        Returns:
            生成的提示文本
        Raises:
            ValueError: If server not connected or prompt call fails.
        """
        if server_name not in self.connections:
            raise ValueError(f"未连接到服务器 '{server_name}'。可用连接: {list(self.connections.keys())}")

        connection = self.connections[server_name]
        if not connection.connected:
            raise ValueError(f"服务器 '{server_name}' 连接已断开。")

        # The actual call logic is in ServerConnection.call_prompt
        return await connection.call_prompt(prompt_name, params)


    async def cleanup(self):
        """清理所有资源，断开所有连接"""
        print("\n正在清理所有客户端连接...")
        # Create a list of names to avoid issues while iterating and modifying the dict
        server_names_to_disconnect = list(self.connections.keys())

        if not server_names_to_disconnect:
            print("没有活动的连接需要清理。")
            return

        cleanup_tasks = []
        for server_name in server_names_to_disconnect:
            # Create cleanup task for each server
            cleanup_tasks.append(
                 asyncio.create_task(self.disconnect_server(server_name))
            )

        # Wait for all disconnect tasks to complete
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Check results for errors
        for i, result in enumerate(results):
             server_name = server_names_to_disconnect[i]
             if isinstance(result, Exception):
                 print(f"清理服务器 '{server_name}' 时发生错误: {result}")
             elif result is False: # disconnect_server returns False on cancellation
                  print(f"清理服务器 '{server_name}' 被取消。")


        # Final confirmation
        self.connections = {} # Ensure dict is empty
        self.current_connection = None
        self.server_config = None
        print("所有客户端连接清理完成。")


    async def interactive_mode(self):
        """运行交互式命令行界面"""
        print("\n欢迎使用 lite_mcp_client 交互模式!")
        self._print_help()

        should_exit = False
        while not should_exit:
            try:
                prompt_str = "命令"
                if self.current_connection:
                    prompt_str += f" ({self.current_connection.name})>"
                else:
                     prompt_str += ">"

                cmd_line = input(f"\n{prompt_str} ").strip()
                if not cmd_line:
                    continue

                parts = cmd_line.split(maxsplit=1)
                command = parts[0].lower()
                args_str = parts[1] if len(parts) > 1 else ""

                # --- Command Handling ---
                if command == 'quit' or command == 'exit':
                    should_exit = True
                    print("正在退出...")
                    # Let the main finally block handle cleanup

                elif command == 'help':
                    self._print_help()

                elif command == 'clear-history':
                    self.history = []
                    print("对话历史已清除。")

                elif command == 'connect':
                    if not args_str:
                        print("用法: connect <服务器名>")
                        self.list_available_servers(self.load_server_config())
                        continue
                    server_name = args_str
                    try:
                        config = self.load_server_config()
                        await self.connect_to_server_by_name(server_name, config)
                    except Exception as e:
                        print(f"连接到 '{server_name}' 时出错: {e}")

                elif command == 'connect-all':
                    try:
                        config = self.load_server_config()
                        await self.connect_all_default_servers(config)
                    except Exception as e:
                        print(f"连接所有默认服务器时出错: {e}")

                elif command == 'disconnect':
                    if not args_str:
                        print("用法: disconnect <服务器名>")
                        self.list_connections()
                        continue
                    server_name = args_str
                    try:
                        await self.disconnect_server(server_name)
                    except Exception as e:
                         print(f"断开 '{server_name}' 时出错: {e}")

                elif command == 'switch':
                    if not args_str:
                        print("用法: switch <服务器名>")
                        self.list_connections()
                        continue
                    server_name = args_str
                    try:
                        await self.switch_connection(server_name)
                    except Exception as e: # switch_connection itself prints errors
                        pass

                elif command == 'connections' or command == 'conn':
                    self.list_connections()

                elif command == 'tools':
                    self.list_tools()

                elif command == 'resources' or command == 'res':
                    self.list_resources()

                elif command == 'prompts':
                    self.list_prompts()

                elif command == 'call':
                    if not args_str:
                        print("用法: call <服务器名.工具名> [参数JSON]")
                        self.list_tools()
                        continue
                    call_parts = args_str.split(maxsplit=1)
                    tool_spec = call_parts[0]
                    params_json = call_parts[1] if len(call_parts) > 1 else "{}"

                    if '.' not in tool_spec:
                        print("错误: 工具名必须包含服务器名，格式: 服务器名.工具名")
                        continue

                    server_name, tool_name = tool_spec.split('.', 1)
                    params = {}
                    try:
                        params = json.loads(params_json)
                        if not isinstance(params, dict):
                            raise ValueError("参数必须是一个JSON对象 (字典)")
                    except json.JSONDecodeError as e:
                        print(f"错误: 无效的JSON参数: {e}")
                        continue
                    except ValueError as e:
                         print(f"错误: {e}")
                         continue

                    print(f"\n正在调用 {server_name}.{tool_name}...")
                    try:
                        result = await self.call_tool_by_server(server_name, tool_name, params)
                        print("\n结果:")
                        try:
                            # Pretty print JSON if possible, otherwise raw string
                            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                        except TypeError:
                             print(result)
                    except ValueError as e: # Specific error from call_tool_by_server
                         print(f"\n调用失败: {e}")
                    except Exception as e:
                        print(f"\n调用时发生意外错误: {e}")
                        # traceback.print_exc() # Uncomment for detailed debugging

                elif command == 'get':
                     if not args_str:
                         print("用法: get <服务器名.资源URI>")
                         self.list_resources()
                         continue
                     res_spec = args_str

                     if '.' not in res_spec:
                         print("错误: 资源URI必须包含服务器名，格式: 服务器名.资源URI")
                         continue

                     server_name, res_uri = res_spec.split('.', 1)

                     print(f"\n正在获取 {server_name}.{res_uri}...")
                     try:
                         result = await self.get_resource_by_server(server_name, res_uri)
                         print("\n获取结果:")
                         print(result)
                     except ValueError as e:
                          print(f"\n获取失败: {e}")
                     except Exception as e:
                         print(f"\n获取时发生意外错误: {e}")

                elif command == 'prompt':
                    if not args_str:
                        print("用法: prompt <服务器名.提示名> [参数JSON]")
                        self.list_prompts()
                        continue
                    prompt_parts = args_str.split(maxsplit=1)
                    prompt_spec = prompt_parts[0]
                    params_json = prompt_parts[1] if len(prompt_parts) > 1 else "{}"

                    if '.' not in prompt_spec:
                        print("错误: 提示名必须包含服务器名，格式: 服务器名.提示名")
                        continue

                    server_name, prompt_name = prompt_spec.split('.', 1)
                    params = {}
                    try:
                        params = json.loads(params_json)
                        if not isinstance(params, dict):
                             raise ValueError("参数必须是一个JSON对象 (字典)")
                    except json.JSONDecodeError as e:
                        print(f"错误: 无效的JSON参数: {e}")
                        continue
                    except ValueError as e:
                         print(f"错误: {e}")
                         continue

                    print(f"\n正在调用提示模板 {server_name}.{prompt_name}...")
                    try:
                        result = await self.call_prompt_by_server(server_name, prompt_name, params)
                        print("\n生成的提示:")
                        print(result)
                    except ValueError as e:
                         print(f"\n调用失败: {e}")
                    except Exception as e:
                        print(f"\n调用时发生意外错误: {e}")

                elif command == 'ask':
                    if not args_str:
                        print("用法: ask <你的自然语言问题>")
                        continue
                    query = args_str
                    print(f"\n正在处理LLM回答: {query}")
                    try:
                        result = await self.smart_query(query)
                        print("\nLLM回答:")
                        print(result)
                    except Exception as e:
                         print(f"\nLLM回答失败: {e}")
                         # traceback.print_exc() # Uncomment for debugging

                else:
                    print(f"未知命令: '{command}'. 输入 'help' 查看可用命令。")

            except KeyboardInterrupt:
                print("\n检测到 Ctrl+C，正在退出...")
                should_exit = True
                # Let the main finally block handle cleanup
            except EOFError: # Handle Ctrl+D or end of input stream
                 print("\n检测到输入结束，正在退出...")
                 should_exit = True
            except Exception as e:
                print(f"\n交互模式中发生错误: {str(e)}")
                import traceback
                traceback.print_exc() # Print detailed traceback for debugging


        print("交互模式结束。")

    def _print_help(self):
        """Prints help message for interactive mode"""
        print("\n可用命令:")
        print("  connect <服务器名>       - 连接到配置文件中定义的指定服务器")
        print("  connect-all             - 连接到配置文件中所有标记为默认的服务器")
        print("  disconnect <服务器名>    - 断开与指定服务器的连接并清理资源")
        print("  switch <服务器名>        - 切换当前活动的服务器连接")
        print("  connections (conn)      - 列出所有当前连接及其状态")
        print("  tools                   - 列出所有已连接服务器的可用工具")
        print("  resources (res)         - 列出所有已连接服务器的可用资源")
        print("  prompts                 - 列出所有已连接服务器的可用提示模板")
        print("  call <srv.tool> [params] - 调用指定服务器上的工具 (params为JSON)")
        print("  get <srv.uri>           - 获取指定服务器上的资源")
        print("  prompt <srv.prompt> [params] - 使用指定服务器上的提示模板 (params为JSON)")
        print("  ask <自然语言问题>        - LLM处理提问，自动选择并调用工具")
        print("  clear-history           - 清除 'ask' 命令的对话历史记录")
        print("  help                    - 显示此帮助信息")
        print("  quit / exit             - 退出程序")


    async def get_resource_by_server(self, server_name: str, resource_uri: str) -> Any:
        """获取指定服务器上的资源

        Args:
            server_name: 服务器名称
            resource_uri: 资源URI

        Returns:
            资源内容
        Raises:
            ValueError: If server not connected or resource fetch fails.
        """
        if server_name not in self.connections:
            raise ValueError(f"未连接到服务器 '{server_name}'。可用连接: {list(self.connections.keys())}")

        connection = self.connections[server_name]
        if not connection.connected:
            raise ValueError(f"服务器 '{server_name}' 连接已断开。")

        return await connection.get_resource(resource_uri)

    def list_tools(self):
        """列出所有已连接服务器的工具"""
        active_connections = {name: conn for name, conn in self.connections.items() if conn.connected}
        if not active_connections:
            print("\n没有活动的服务器连接。")
            return

        print("\n所有活动服务器的可用工具:")
        found_any = False
        for server_name, conn in active_connections.items():
            print(f"\n## 服务器: {server_name}")
            if not conn.tools_cache:
                print("  (没有可用的工具)")
                continue

            found_any = True
            for i, tool in enumerate(conn.tools_cache, 1):
                print(f"\n  {i}. {tool['name']}")
                print(f"     描述: {tool.get('description', '无')}")
                print(f"     参数:")
                schema = tool.get('schema', {})
                if isinstance(schema, dict) and schema.get('properties'):
                    required_params = schema.get('required', [])
                    for param_name, param_info in schema['properties'].items():
                         if isinstance(param_info, dict):
                             param_type = param_info.get('type', 'any')
                             param_desc = param_info.get('description', '')
                             required_star = "*" if param_name in required_params else ""
                             print(f"       - {param_name}{required_star} ({param_type}): {param_desc}")
                         else:
                              print(f"       - {param_name}: (无效的参数信息)")
                else:
                    print("       (无参数或无详细信息)")
        if not found_any:
             print("\n所有活动服务器均未报告可用工具。")


    def list_resources(self):
        """列出所有已连接服务器的资源"""
        active_connections = {name: conn for name, conn in self.connections.items() if conn.connected}
        if not active_connections:
            print("\n没有活动的服务器连接。")
            return

        print("\n所有活动服务器的可用资源:")
        found_any = False
        for server_name, conn in active_connections.items():
            print(f"\n## 服务器: {server_name}")
            if not conn.resources_cache:
                print("  (没有可用的资源)")
                continue

            found_any = True
            for i, resource in enumerate(conn.resources_cache, 1):
                print(f"  {i}. {resource['uri']}")
                if resource.get('description'):
                    print(f"     描述: {resource['description']}")
        if not found_any:
             print("\n所有活动服务器均未报告可用资源。")


    def list_prompts(self):
        """列出所有已连接服务器的提示模板"""
        active_connections = {name: conn for name, conn in self.connections.items() if conn.connected}
        if not active_connections:
            print("\n没有活动的服务器连接。")
            return

        print("\n所有活动服务器的可用提示模板:")
        found_any = False
        for server_name, conn in active_connections.items():
            print(f"\n## 服务器: {server_name}")
            if not conn.prompts_cache:
                print("  (没有可用的提示模板)")
                continue

            found_any = True
            for i, prompt in enumerate(conn.prompts_cache, 1):
                print(f"\n  {i}. {prompt['name']}")
                print(f"     描述: {prompt.get('description', '无')}")
                print(f"     参数:")
                schema = prompt.get('schema', {})
                if isinstance(schema, dict) and schema.get('properties'):
                    required_params = schema.get('required', [])
                    for param_name, param_info in schema['properties'].items():
                         if isinstance(param_info, dict):
                            param_type = param_info.get('type', 'any')
                            param_desc = param_info.get('description', '')
                            required_star = "*" if param_name in required_params else ""
                            print(f"       - {param_name}{required_star} ({param_type}): {param_desc}")
                         else:
                              print(f"       - {param_name}: (无效的参数信息)")

                else:
                     print("       (无参数或无详细信息)")
        if not found_any:
             print("\n所有活动服务器均未报告可用提示模板。")


    def list_connections(self):
        """列出所有当前的连接状态"""
        if not self.connections:
            print("\n当前没有服务器连接记录。")
            return

        print("\n当前连接状态:")
        for server_name, conn in self.connections.items():
            status = "已连接" if conn.connected else "已断开"
            current_marker = " (当前活动)" if conn is self.current_connection else ""
            print(f"  - {server_name}: {status}{current_marker}")

        if not self.current_connection and self.connections:
             print("\n注意：当前没有设置活动服务器。使用 'switch <服务器名>' 选择一个。")


        # switch_to_server is replaced by the async switch_connection method


    async def connect_all_default_servers(self, config: Dict) -> bool:
        """连接到所有默认服务器

        Args:
            config: 配置字典

        Returns:
            是否至少成功连接一个服务器
        """
        # Get default server names - handle string or list
        default_server_names = config.get('default_server', [])
        if isinstance(default_server_names, str):
            default_server_names = [default_server_names]

        if not default_server_names:
            print("配置中未指定默认服务器。")
            return False

        print(f"\n正在尝试连接所有默认服务器: {', '.join(default_server_names)}")
        overall_success = False
        connect_tasks = []

        # Find configurations first
        servers_to_connect = []
        all_server_configs = {srv.get('name'): srv for srv in config.get('mcp_servers', [])}

        for server_name in default_server_names:
             if server_name in all_server_configs:
                 servers_to_connect.append(all_server_configs[server_name])
             else:
                  print(f"警告: 找不到默认服务器 '{server_name}' 的配置，将跳过。")


        if not servers_to_connect:
             print("没有找到有效的默认服务器配置。")
             return False


        # Create connection tasks
        for server_config in servers_to_connect:
             server_name = server_config['name']
             # Skip if already connected and active
             if server_name in self.connections and self.connections[server_name].connected:
                 print(f"服务器 '{server_name}' 已连接，跳过。")
                 overall_success = True # Count existing connections as success
                 continue
             # Create task to connect
             connect_tasks.append(
                 asyncio.create_task(self.connect_to_server_config(server_config))
             )

        # Wait for all connection tasks to complete
        if connect_tasks:
            results = await asyncio.gather(*connect_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                # Find corresponding server name (needs careful indexing or storing tuples)
                # This part is tricky as tasks don't directly carry the server name.
                # Let's rely on the output from connect_to_server_config for now.
                if isinstance(result, Exception):
                    # Error should have been printed within connect_to_server_config
                    print(f"连接默认服务器时发生异常: {result}")
                elif result is True:
                    overall_success = True
                # False means connection failed, error printed inside connect method


        if not overall_success:
            print("未能成功连接到任何默认服务器。")
        else:
             print("默认服务器连接尝试完成。")

        return overall_success


    async def server_management_mode(self, config_path: str) -> bool:
        """服务器管理模式 (交互式，用于初始连接)

        Args:
            config_path: 配置文件路径

        Returns:
            是否成功连接到至少一个服务器并退出管理模式
        """
        try:
            config = self.load_server_config(config_path)
        except ValueError as e:
            print(f"错误: {e}")
            return False # Cannot proceed without config

        print("\n--- 服务器管理模式 ---")
        self.list_available_servers(config)

        while True:
            print("\n管理命令: connect <编号/名称> | connect-all | list | quit")
            try:
                 cmd_line = input("管理模式> ").strip()
                 if not cmd_line: continue

                 parts = cmd_line.split(maxsplit=1)
                 command = parts[0].lower()
                 args_str = parts[1] if len(parts) > 1 else ""

                 if command == 'quit':
                     print("退出管理模式，未连接服务器。")
                     return False
                 elif command == 'list':
                     self.list_available_servers(config)
                 elif command == 'connect-all':
                     print("尝试连接所有默认服务器...")
                     if await self.connect_all_default_servers(config):
                         print("至少连接了一个默认服务器。退出管理模式。")
                         return True
                     else:
                         print("未能连接到任何默认服务器。")
                 elif command == 'connect':
                     if not args_str:
                         print("请提供服务器编号或名称。")
                         continue
                     server_spec = args_str
                     server_to_connect = None
                     server_name_to_connect = None

                     # Try interpreting as number first
                     try:
                         server_idx = int(server_spec) - 1
                         if 0 <= server_idx < len(config.get('mcp_servers', [])):
                             server_to_connect = config['mcp_servers'][server_idx]
                             server_name_to_connect = server_to_connect.get('name')
                         else:
                             print(f"无效的服务器编号: {server_spec}")
                     except ValueError:
                         # Not a number, treat as name
                         server_name_to_connect = server_spec
                         # Find config by name
                         for srv in config.get('mcp_servers', []):
                             if srv.get('name') == server_name_to_connect:
                                 server_to_connect = srv
                                 break
                         if not server_to_connect:
                              print(f"找不到名为 '{server_name_to_connect}' 的服务器配置。")


                     # If a valid server config was found (by number or name)
                     if server_to_connect and server_name_to_connect:
                          print(f"尝试连接到 '{server_name_to_connect}'...")
                          if await self.connect_to_server_config(server_to_connect):
                              print(f"成功连接到 '{server_name_to_connect}'。退出管理模式。")
                              return True
                          else:
                              print(f"连接到 '{server_name_to_connect}' 失败。")
                 else:
                     print(f"未知管理命令: '{command}'")

            except KeyboardInterrupt:
                 print("\n操作中断。退出管理模式。")
                 return False
            except EOFError:
                 print("\n输入结束。退出管理模式。")
                 return False
            except Exception as e:
                 print(f"\n管理模式中发生错误: {e}")

    def _extract_json_from_text(self, text: str) -> Tuple[str, Dict]:
        """从文本中提取JSON内容"""
        # 在处理前先检查和清理输入文本
        if not text or not isinstance(text, str):
            return "", {}
        
        # 如果文本包含多余的反斜杠，尝试消除它们
        if '\\' in text and '\\\\' not in text:
            cleaned_text = text.replace('\\"', '"')
            try:
                parsed_json = json.loads(cleaned_text)
                return cleaned_text, parsed_json
            except json.JSONDecodeError:
                pass  # 继续尝试其他策略
        
        # 尝试不同的提取策略
        strategies = [
            # 查找代码块中的JSON
            lambda t: re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', t),
            # 查找{}包围的内容
            lambda t: re.search(r'({[\s\S]*?})', t),
            # 查找第一个{到最后一个}
            lambda t: re.search(r'({[\s\S]*})', t) if '{' in t and '}' in t else None
        ]
        
        json_text = ""
        parsed_json = {}
        
        for strategy in strategies:
            match = strategy(text)
            if match:
                json_text = match.group(1).strip()
                try:
                    parsed_json = json.loads(json_text)
                    return json_text, parsed_json
                except json.JSONDecodeError:
                    # 尝试清理JSON文本
                    cleaned_json = self._clean_json_text(json_text)
                    try:
                        parsed_json = json.loads(cleaned_json)
                        return cleaned_json, parsed_json
                    except json.JSONDecodeError:
                        # 继续尝试下一个策略
                        continue
        
        # 所有策略都失败，尝试整个文本
        try:
            parsed_json = json.loads(text)
            return text, parsed_json
        except json.JSONDecodeError:
            # 尝试清理整个文本
            cleaned_text = self._clean_json_text(text)
            try:
                parsed_json = json.loads(cleaned_text)
                return cleaned_text, parsed_json
            except json.JSONDecodeError:
                # 所有尝试都失败
                return json_text, {}

    def _clean_json_text(self, text: str) -> str:
        """清理JSON文本，移除可能导致解析错误的内容
        
        Args:
            text: 需要清理的JSON文本
            
        Returns:
            清理后的JSON文本
        """
        # 移除可能的Markdown格式
        text = re.sub(r'```.*\n|```', '', text)
        
        # 移除多余的空格和换行
        text = text.strip()
        
        # 替换不规范的JSON语法
        text = re.sub(r'(\w+):', r'"\1":', text)  # 将 key: 替换为 "key":
        text = re.sub(r',\s*}', '}', text)        # 移除对象末尾的逗号
        text = re.sub(r',\s*]', ']', text)        # 移除数组末尾的逗号
        
        # 处理可能的多行字符串格式问题
        text = re.sub(r'"""(.*?)"""', r'"\1"', text, flags=re.DOTALL)
        text = re.sub(r"'''(.*?)'''", r'"\1"', text, flags=re.DOTALL)
        
        return text

    async def _call_llm_for_next_step(self, transcript, available_tools, available_prompts, available_resources):
        """调用 LLM 确定下一步行动"""
        if not self.llm:
            return {"action": "error", "error": "错误: LLM 未初始化，无法执行LLM回答。"}
        
        # 格式化工具和提示信息
        tools_info = self._format_tools_for_llm()
        prompts_info = self._format_prompts_for_llm()
        
        # 准备系统提示 - 注意所有JSON示例中的大括号要用双大括号转义
        system_prompt = f"""你是一个全能助手，可以在有需要时使用以下工具来回答用户问题。

--- 可用工具 ---
{tools_info}
--- 可用提示模板 ---
{prompts_info}
---

你的思考和行动步骤:
1.  **分析请求:** 理解用户的问题。
2.  **规划:** 决定是否需要调用工具或提示模板。如果需要，选择最合适的，并确定所需的参数。参数名旁边带 * 号的是必需的。
3.  **行动:**
    *   **调用工具:** 如果需要调用工具，使用以下精确格式（包括代码块标记）。一次只能调用一个工具。
        ```json
        ACTION: call_tool
        {{
          "tool": "服务器名.工具名",
          "params": {{
            "参数名1": "参数值1",
            "参数名2": "参数值2"
          }}
        }}
        ```
    *   **调用提示:** 如果需要使用提示模板，使用以下精确格式。
        ```json
        ACTION: call_prompt
        {{
          "prompt": "服务器名.提示名",
          "params": {{
            "参数名1": "参数值1"
          }}
        }}
        ```
    *   **回答:** 当你收集到足够的信息或者认为不需要调用工具时，直接回答用户。使用以下格式：
        ```text
        ACTION: final_answer
        <这里是给用户的最终回答>
        ```
4.  **迭代:** 在收到工具或提示的调用结果后，重复步骤 1-3，直到任务完成。

**重要规则:**
*   **必需参数:** 确保为标记为 * 的必需参数提供值。
*   **一次一个 Action:** 每个回复只能包含一个 `ACTION`。
*   **精确格式:** 严格遵守 `ACTION` 的 JSON 格式。JSON 必须有效。
*   **服务器名:** 必须在 `tool` 或 `prompt` 字段中包含服务器名 (例如 `服务器A.get_news`)。
*   **等待结果:** 在每次调用工具或提示后，我会提供结果，请等待结果再进行下一步。
*   **最终答案:** 只有在你完成所有必要步骤并准备好最终答案时，才使用 `ACTION: final_answer`。
"""
        
        # 准备消息列表
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 添加对话历史
        for msg in transcript:
            messages.append(msg)
            
        # 调用LLM
        llm_response = None
        try:
            # 转换消息格式以适应不同的LLM接口
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            response = await self.llm.ainvoke(langchain_messages)
            llm_content = response.content.strip()
            
            # 保存原始响应
            llm_response = llm_content
            
            # 解析响应，查找ACTION标记
            try:
                # 使用组合标志
                action_match = re.search(r'ACTION:\s*(.*?)(?:\n|$)', llm_content, re.IGNORECASE)
                if not action_match:
                    print("未找到ACTION标记，默认为最终答案")
                    return {
                        "action": "final_answer",
                        "content": llm_content,
                        "raw_response": llm_content
                    }
                # ...其他处理...
            except Exception as regex_err:
                print(f"处理ACTION时出错: {regex_err}")
                # 提供一个友好的错误响应
                return {
                    "action": "error",
                    "error": f"解析LLM响应时出错: {regex_err}",
                    "raw_response": llm_content
                }
            
            action_type = action_match.group(1).strip().lower()
            
            if "final_answer" in action_type:
                # 提取最终答案内容
                final_answer_match = re.search(r'ACTION:\s*final_answer\s*(.*)', llm_content, re.DOTALL | re.IGNORECASE)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip()
                else:
                    final_answer = llm_content.replace("ACTION: final_answer", "").strip()
                
                return {
                    "action": "final_answer",
                    "content": final_answer,
                    "raw_response": llm_content
                }
            
            elif "call_tool" in action_type:
                # 从响应中提取JSON
                try:
                    json_text, json_data = self._extract_json_from_text(llm_content)
                except Exception as json_err:
                    print(f"JSON提取错误: {json_err}")
                    # 提供一个友好的错误响应
                    return {
                        "action": "error",
                        "error": f"从LLM响应中提取JSON时出错: {json_err}",
                        "raw_response": llm_content
                    }

                if not json_data:
                    # 尝试更强力的JSON修复
                    try:
                        # 查找ACTION后的第一个{和最后一个}之间的内容
                        json_start = llm_content.find('{', llm_content.find('ACTION'))
                        json_end = llm_content.rfind('}') + 1
                        if json_start > 0 and json_end > json_start:
                            json_text = llm_content[json_start:json_end]
                            # 替换不合法的JSON格式
                            json_text = re.sub(r'(?m)^\s*(\w+):', r'"\1":', json_text)
                            json_text = re.sub(r',\s*}', '}', json_text)
                            json_text = re.sub(r',\s*]', ']', json_text)
                            try:
                                json_data = json.loads(json_text)
                            except:
                                # 最后尝试
                                json_lines = []
                                capture = False
                                for line in llm_content.split('\n'):
                                    if '{' in line:
                                        capture = True
                                    if capture:
                                        json_lines.append(line)
                                    if '}' in line and capture:
                                        break
                                if json_lines:
                                    json_text = '\n'.join(json_lines)
                                    json_text = re.sub(r'(?m)^\s*(\w+):', r'"\1":', json_text)
                                    json_text = re.sub(r',\s*}', '}', json_text)
                                    json_text = re.sub(r',\s*]', ']', json_text)
                                    try:
                                        json_data = json.loads(json_text)
                                    except:
                                        pass
                    except:
                        pass

                # 如果仍然没有有效JSON
                if not json_data:
                    json_data = self._emergency_parse_json(llm_content)
                    
                    if not json_data:
                        return {
                            "action": "error",
                            "error": "无法从LLM响应中提取有效的JSON数据",
                            "raw_response": llm_content
                        }
                    else:
                        print(f"JSON解析成功: {json_data}")

                tool_spec = json_data.get("tool")
                params = json_data.get("params", {})
                
                if not tool_spec:
                    return {
                        "action": "error",
                        "error": "工具调用缺少工具名称",
                        "raw_response": llm_content
                    }
                
                # 解析服务器名和工具名
                if '.' in tool_spec:
                    server_name, tool_name = tool_spec.split('.', 1)
                else:
                    if not self.connections:
                        return {
                            "action": "error",
                            "error": "未指定服务器名，且没有连接到任何服务器",
                            "raw_response": llm_content
                        }
                    server_name = next(iter(self.connections.keys()))
                    tool_name = tool_spec
                
                return {
                    "action": "call_tool",
                    "server": server_name,
                    "tool": tool_name,
                    "params": params,
                    "raw_response": llm_content
                }
            
            elif "call_prompt" in action_type:
                # 从响应中提取JSON
                try:
                    json_text, json_data = self._extract_json_from_text(llm_content)
                except Exception as json_err:
                    print(f"JSON提取错误: {json_err}")
                    # 提供一个友好的错误响应
                    return {
                        "action": "error",
                        "error": f"从LLM响应中提取JSON时出错: {json_err}",
                        "raw_response": llm_content
                    }

                if not json_data:
                    try:
                        # 查找ACTION后的第一个{和最后一个}之间的内容
                        json_start = llm_content.find('{', llm_content.find('ACTION'))
                        json_end = llm_content.rfind('}') + 1
                        if json_start > 0 and json_end > json_start:
                            json_text = llm_content[json_start:json_end]
                            # 替换不合法的JSON格式
                            json_text = re.sub(r'(?m)^\s*(\w+):', r'"\1":', json_text)
                            json_text = re.sub(r',\s*}', '}', json_text)
                            json_text = re.sub(r',\s*]', ']', json_text)
                            try:
                                json_data = json.loads(json_text)
                            except:
                                # 最后尝试
                                json_lines = []
                                capture = False
                                for line in llm_content.split('\n'):
                                    if '{' in line:
                                        capture = True
                                    if capture:
                                        json_lines.append(line)
                                    if '}' in line and capture:
                                        break
                                if json_lines:
                                    json_text = '\n'.join(json_lines)
                                    json_text = re.sub(r'(?m)^\s*(\w+):', r'"\1":', json_text)
                                    json_text = re.sub(r',\s*}', '}', json_text)
                                    json_text = re.sub(r',\s*]', ']', json_text)
                                    try:
                                        json_data = json.loads(json_text)
                                    except:
                                        pass
                    except:
                        pass

                # 如果仍然没有有效JSON
                if not json_data:
                    # 最后尝试使用紧急解析
                    print("尝试使用紧急JSON解析方法...")
                    json_data = self._emergency_parse_json(llm_content)
                    
                    if not json_data:
                        return {
                            "action": "error",
                            "error": "无法从LLM响应中提取有效的JSON数据",
                            "raw_response": llm_content
                        }
                    else:
                        print(f"紧急JSON解析成功: {json_data}")

                prompt_spec = json_data.get("prompt")
                params = json_data.get("params", {})
                
                if not prompt_spec:
                    return {
                        "action": "error",
                        "error": "提示调用缺少提示模板名称",
                        "raw_response": llm_content
                    }
                
                # 解析服务器名和提示名
                if '.' in prompt_spec:
                    server_name, prompt_name = prompt_spec.split('.', 1)
                else:
                    if not self.connections:
                        return {
                            "action": "error",
                            "error": "未指定服务器名，且没有连接到任何服务器",
                            "raw_response": llm_content
                        }
                    server_name = next(iter(self.connections.keys()))
                    prompt_name = prompt_spec
                
                return {
                    "action": "call_prompt",
                    "server": server_name,
                    "prompt": prompt_name,
                    "params": params,
                    "raw_response": llm_content
                }
            
            else:
                # 未知的动作类型
                return {
                    "action": "error",
                    "error": f"未知的动作类型: {action_type}",
                    "raw_response": llm_content
                }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "action": "error",
                "error": f"处理LLM响应时出错: {str(e)}",
                "raw_response": llm_response or "LLM调用失败"
            }

    def _emergency_parse_json(self, text: str) -> Dict:
        """当正常JSON解析失败时的紧急解析方法
        
        Args:
            text: 可能包含JSON的文本
            
        Returns:
            解析的JSON字典或空字典
        """
        try:
            # 1. 尝试查找所有 { } 块
            brace_pattern = re.compile(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}')
            matches = brace_pattern.findall(text)
            
            for match in matches:
                try:
                    # 尝试直接解析
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        return parsed
                except:
                    # 尝试清理后解析
                    cleaned = re.sub(r'(\w+):', r'"\1":', match)  # key -> "key"
                    cleaned = re.sub(r',\s*}', '}', cleaned)      # remove trailing commas
                    try:
                        parsed = json.loads(cleaned)
                        if isinstance(parsed, dict) and len(parsed) > 0:
                            return parsed
                    except:
                        continue  # 尝试下一个匹配
            
            # 2. 如果没有找到有效的JSON，尝试截取 "tool" 或 "prompt" 相关内容
            tool_pattern = re.compile(r'"tool"\s*:\s*"([^"]+)"')
            tool_match = tool_pattern.search(text)
            
            prompt_pattern = re.compile(r'"prompt"\s*:\s*"([^"]+)"')
            prompt_match = prompt_pattern.search(text)
            
            params_pattern = re.compile(r'"params"\s*:\s*({[^{}]*(?:{[^{}]*}[^{}]*)*})')
            params_match = params_pattern.search(text)
            
            result = {}
            
            if tool_match:
                result["tool"] = tool_match.group(1)
            elif prompt_match:
                result["prompt"] = prompt_match.group(1)
                
            if params_match:
                try:
                    params_text = params_match.group(1)
                    # 尝试清理并解析参数
                    cleaned_params = re.sub(r'(\w+):', r'"\1":', params_text)
                    cleaned_params = re.sub(r',\s*}', '}', cleaned_params)
                    params = json.loads(cleaned_params)
                    result["params"] = params
                except:
                    result["params"] = {}
            
            return result
        except Exception as e:
            print(f"紧急JSON解析也失败: {e}")
            return {}


async def main():
    """主函数"""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='lite_mcp_client: 一个通用的 MCP 命令行客户端。',
        epilog="如果没有提供操作参数 (如 --query, --call, --interactive), "
               "程序将尝试连接默认服务器并进入交互模式。"
    )
    parser.add_argument(
        '--config', '-c',
        default=DEFAULT_CONFIG_PATH,
        help=f'指定配置文件的路径 (默认: {DEFAULT_CONFIG_PATH})'
    )

    # Connection options
    conn_group = parser.add_argument_group('连接选项')
    conn_group.add_argument(
        '--server',
        metavar='服务器名',
        help='启动时连接到指定的服务器'
    )
    conn_group.add_argument(
        '--connect-all',
        action='store_true',
        help='启动时连接到配置文件中所有标记为默认的服务器'
    )

    # Action options (mutually exclusive group ideally, but complex with interactive fallback)
    action_group = parser.add_argument_group('操作选项 (如果提供，将执行并退出)')
    action_group.add_argument(
        '--query', '--ask', # Allow both --query and --ask
        metavar='自然语言回答',
        help='使用 LLM LLM处理回答并获取结果'
    )
    action_group.add_argument(
        '--call',
        metavar='服务器名.工具名',
        help='直接调用指定的工具'
    )
    action_group.add_argument(
        '--prompt',
        metavar='服务器名.提示名',
        help='直接调用指定的提示模板'
    )
    action_group.add_argument(
        '--get',
        metavar='服务器名.资源URI',
        help='直接获取指定的资源'
    )
    action_group.add_argument(
        '--params',
        metavar='JSON字符串',
        help='为 --call 或 --prompt 提供参数 (必须是有效的JSON对象)'
    )

    # Mode option
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='强制进入交互模式，即使提供了其他操作参数'
    )

    # Positional argument for simple queries (must be last)
    parser.add_argument(
        'direct_query',
        nargs='?', # Optional positional argument
        metavar='回答文本',
        help='直接提供自然语言回答 (等同于 --query，不需要标志)'
    )

    args = parser.parse_args()

    # --- Client Initialization ---
    client = GenericMCPClient()
    initial_connection_made = False
    exit_code = 0 # Default exit code

    try:
        # --- Load Config ---
        try:
             config = client.load_server_config(args.config)
        except ValueError as e:
             print(f"错误: {e}")
             sys.exit(1) # Exit if config is fundamentally broken

        # --- Initial Connection Logic ---
        if args.server:
            print(f"尝试连接指定服务器: {args.server}")
            initial_connection_made = await client.connect_to_server_by_name(args.server, config)
        elif args.connect_all:
             print("尝试连接所有默认服务器...")
             initial_connection_made = await client.connect_all_default_servers(config)
        elif not any([args.query, args.direct_query, args.call, args.prompt, args.get, args.interactive]):
             # Default behavior: connect to defaults if no action/interactive specified
             print("未指定操作，尝试连接默认服务器...")
             initial_connection_made = await client.connect_all_default_servers(config)


        # --- Action Execution (if not forced interactive) ---
        action_performed = False
        if not args.interactive:
            query_to_run = args.query or args.direct_query
            if query_to_run:
                action_performed = True
                print(f"\n执行LLM回答: {query_to_run}")
                if not client.connections: # Connect if needed
                     print("LLM回答需要连接，尝试连接默认服务器...")
                     await client.connect_all_default_servers(config)
                if not client.connections:
                     print("错误: 无法连接到任何服务器以执行回答。")
                     exit_code = 1
                else:
                     result = await client.smart_query(query_to_run)
                     print("\n回答结果:")
                     print(result)


            elif args.call:
                 action_performed = True
                 if not args.call or '.' not in args.call:
                     print("错误: --call 参数需要格式 '服务器名.工具名'")
                     exit_code = 1
                 else:
                     server_name, tool_name = args.call.split('.', 1)
                     params = {}
                     if args.params:
                         try:
                             params = json.loads(args.params)
                             if not isinstance(params, dict): raise ValueError("参数必须是JSON对象")
                         except (json.JSONDecodeError, ValueError) as e:
                             print(f"错误: 无效的 --params JSON: {e}")
                             exit_code = 1
                     if exit_code == 0:
                          print(f"\n执行工具调用: {server_name}.{tool_name}")
                          if server_name not in client.connections: # Connect if needed
                               print(f"需要连接到 '{server_name}'...")
                               await client.connect_to_server_by_name(server_name, config)
                          try:
                               result = await client.call_tool_by_server(server_name, tool_name, params)
                               print("\n调用结果:")
                               try:
                                   print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                               except TypeError:
                                    print(result)
                          except Exception as e:
                               print(f"\n工具调用失败: {e}")
                               exit_code = 1


            elif args.prompt:
                 action_performed = True
                 if not args.prompt or '.' not in args.prompt:
                     print("错误: --prompt 参数需要格式 '服务器名.提示名'")
                     exit_code = 1
                 else:
                     server_name, prompt_name = args.prompt.split('.', 1)
                     params = {}
                     if args.params:
                          try:
                             params = json.loads(args.params)
                             if not isinstance(params, dict): raise ValueError("参数必须是JSON对象")
                          except (json.JSONDecodeError, ValueError) as e:
                             print(f"错误: 无效的 --params JSON: {e}")
                             exit_code = 1
                     if exit_code == 0:
                          print(f"\n执行提示调用: {server_name}.{prompt_name}")
                          if server_name not in client.connections:
                               print(f"需要连接到 '{server_name}'...")
                               await client.connect_to_server_by_name(server_name, config)
                          try:
                               result = await client.call_prompt_by_server(server_name, prompt_name, params)
                               print("\n调用结果:")
                               print(result)
                          except Exception as e:
                               print(f"\n提示调用失败: {e}")
                               exit_code = 1


            elif args.get:
                 action_performed = True
                 if not args.get or '.' not in args.get:
                     print("错误: --get 参数需要格式 '服务器名.资源URI'")
                     exit_code = 1
                 else:
                     server_name, resource_uri = args.get.split('.', 1)
                     print(f"\n执行资源获取: {server_name}.{resource_uri}")
                     if server_name not in client.connections:
                          print(f"需要连接到 '{server_name}'...")
                          await client.connect_to_server_by_name(server_name, config)
                     try:
                          result = await client.get_resource_by_server(server_name, resource_uri)
                          print("\n获取结果:")
                          print(result)
                     except Exception as e:
                          print(f"\n资源获取失败: {e}")
                          exit_code = 1


        # --- Enter Interactive Mode? ---
        # Enter if:
        # 1. Explicitly requested (--interactive)
        # 2. No action was performed via command line args
        if args.interactive or not action_performed:
            # If no initial connection was made (e.g., only --interactive flag given),
            # maybe enter server management mode first? Or just start interactive.
            if not client.connections:
                print("\n未连接到任何服务器。")
                # Optionally start management mode here:
                # managed_to_connect = await client.server_management_mode(args.config)
                # if not managed_to_connect:
                #     print("未能连接到服务器，退出。")
                # else:
                #     await client.interactive_mode()

                # Simpler: Just start interactive mode, user needs to 'connect'
                await client.interactive_mode()
            else:
                 await client.interactive_mode()


    except KeyboardInterrupt:
         print("\n操作被用户中断。")
         exit_code = 130 # Standard exit code for Ctrl+C
    except Exception as e:
         print(f"\n发生未处理的主程序错误: {str(e)}")
         import traceback
         traceback.print_exc()
         exit_code = 1
    finally:
        # --- Cleanup ---
        print("\n正在关闭客户端并清理连接...")
        try:
            # Use wait_for for the overall client cleanup
            await asyncio.wait_for(client.cleanup(), timeout=5.0) # Adjust timeout
            # print("客户端清理完成。")
        except asyncio.TimeoutError:
            print("警告: 客户端清理超时。可能存在未释放的资源。")
        except Exception as cleanup_err:
            print(f"清理连接时出错: {cleanup_err}")
            # traceback.print_exc() # Uncomment for debugging cleanup

        # --- Final Task Cancellation (Aggressive) ---
        # Sometimes needed if background tasks or subprocesses hang
        # print("正在取消所有剩余任务...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
             for task in tasks:
                 task.cancel()
             try:
                 # Wait briefly for cancellation to be processed
                 await asyncio.wait(tasks, timeout=1.0, return_when=asyncio.ALL_COMPLETED)
                #  print("剩余任务已取消。")
             except asyncio.TimeoutError:
                  print("警告: 等待任务取消超时。")
             except Exception as task_cancel_err:
                  print(f"取消剩余任务时出错: {task_cancel_err}")
        else:
             print("没有剩余任务需要取消。")


        print(f"程序终止。退出代码: {exit_code}")
        # Use os._exit for a more forceful exit if graceful shutdown is problematic
        # This skips further Python cleanup but ensures termination. Use cautiously.
        os._exit(exit_code)


if __name__ == "__main__":
    # Setup basic logging if needed
    # import logging
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # Run the main async function
    # Standard asyncio run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
         # This might catch the KeyboardInterrupt before main's finally block
         # depending on timing, but the finally block should still execute.
         print("\n程序被中断 (asyncio.run)")
    except Exception as final_err:
         # Catch errors occurring directly within asyncio.run setup/teardown
         print(f"\nAsyncio 运行期间发生顶层错误: {final_err}")
         import traceback
         traceback.print_exc()
         os._exit(1) # Force exit on top-level async error

    # The os._exit() in main's finally block will usually prevent code here from running
    # print("Asyncio.run 已完成。")

