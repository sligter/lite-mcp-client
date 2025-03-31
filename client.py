import asyncio
import sys
import json
import os
from typing import Optional, Dict, List, Any, Tuple, Set
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
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
        self.exit_stack = AsyncExitStack()
        self.tools_cache = []
        self.resources_cache = []
        self.prompts_cache = []
        self.stdio = None
        self.write = None
        self.name = config.get('name', '未命名服务器')
        self.connected = False
        self.subprocess = None  # 跟踪子进程
        self.background_tasks = set()  # 跟踪后台任务
    
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
                raise ValueError(f"不支持的服务器类型: {server_type}")
        except Exception as e:
            print(f"连接到服务器 '{self.name}' 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _connect_stdio(self) -> bool:
        """连接到STDIO类型的服务器
        
        Returns:
            连接是否成功
        """
        config = self.config
        command = config.get('command')
        if not command:
            raise ValueError("STDIO配置中缺少'command'")
            
        args = config.get('args', [])
        env = config.get('env')
        
        print(f"正在连接到STDIO服务器: {command} {' '.join(args)}")
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        
        try:
            # 创建独立的异步上下文，避免与其他连接共用
            my_exit_stack = AsyncExitStack()
            
            # 使用专用上下文创建stdio客户端
            stdio_transport = await my_exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            
            # 如果可能，尝试获取子进程引用
            if hasattr(self.stdio, '_process') and self.stdio._process:
                self.subprocess = self.stdio._process
            
            # 创建会话
            self.session = await my_exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            # 初始化会话，但添加超时
            try:
                await asyncio.wait_for(self.session.initialize(), timeout=5.0)
            except asyncio.TimeoutError:
                print(f"初始化会话超时，但将继续尝试")
            
            # 保存exit_stack
            self.exit_stack = my_exit_stack
            
            # 尝试刷新服务器信息，但即使失败也继续
            try:
                await asyncio.wait_for(self.refresh_server_info(), timeout=3.0)
            except Exception as e:
                print(f"警告: 刷新服务器信息时出错: {str(e)}")
                print("服务器连接已建立，但某些功能可能不可用")
            
            self.connected = True
            return True
        except Exception as e:
            print(f"连接到STDIO服务器失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 如果已创建资源，确保它们被清理
            if hasattr(self, 'exit_stack') and self.exit_stack:
                try:
                    # 尝试安全关闭exit_stack
                    await asyncio.wait_for(self.exit_stack.aclose(), timeout=1.0)
                except Exception:
                    pass  # 忽略清理过程中的异常
                
                self.exit_stack = AsyncExitStack()
            
            # 重置所有状态
            self.session = None
            self.stdio = None
            self.write = None
            self.subprocess = None
            self.connected = False
            
            return False
    
    async def _connect_sse(self) -> bool:
        """连接到SSE类型的服务器
        
        Returns:
            连接是否成功
        """
        config = self.config
        url = config.get('url')
        if not url:
            raise ValueError("SSE配置中缺少'url'")
            
        headers = config.get('headers', {})
        
        print(f"正在连接到SSE服务器: {url}")
        
        try:
            from mcp.client.sse import SSEServerParameters, sse_client
            
            server_params = SSEServerParameters(
                url=url,
                headers=headers
            )
            
            sse_transport = await self.exit_stack.enter_async_context(sse_client(server_params))
            self.session = await self.exit_stack.enter_async_context(ClientSession(sse_transport))
            
            await self.session.initialize()
            await self.refresh_server_info()
            self.connected = True
            return True
        except ImportError:
            print("缺少SSE客户端支持，请安装必要的依赖")
            return False
    
    async def refresh_server_info(self):
        """刷新服务器提供的工具、资源和提示信息"""
        if not self.session:
            return
        
        server_info = []
        
        # 获取工具列表
        try:
            response = await self.session.list_tools()
            self.tools_cache = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema
                } for tool in response.tools
            ]
            server_info.append(f"{len(self.tools_cache)} 个工具")
        except Exception as e:
            print(f"获取工具列表时出错: {str(e)}")
            self.tools_cache = []
            server_info.append("工具列表不可用")
        
        # 获取资源列表
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
            print(f"获取资源列表时出错: {str(e)}")
            self.resources_cache = []
            server_info.append("资源列表不可用")
        
        # 获取提示列表
        try:
            response = await self.session.list_prompts()
            self.prompts_cache = []
            
            # 处理提示对象的属性
            for prompt in response.prompts:
                prompt_info = {
                    "name": prompt.name,
                    "description": prompt.description,
                }
                
                # 检查prompt对象是否有input_schema或schema属性
                schema_value = None
                if hasattr(prompt, 'input_schema'):
                    schema_value = prompt.input_schema
                elif hasattr(prompt, 'schema'):
                    schema_value = prompt.schema
                elif hasattr(prompt, 'inputSchema'):
                    schema_value = prompt.inputSchema
                
                # 处理schema可能是方法的情况
                if callable(schema_value):
                    try:
                        # 尝试优先使用model_json_schema方法（Pydantic v2），如果不可用则回退到schema方法
                        if hasattr(schema_value, 'model_json_schema'):
                            schema_value = schema_value.model_json_schema()
                        else:
                            schema_value = schema_value()
                    except Exception as e:
                        print(f"调用schema方法失败: {e}")
                        schema_value = {}
                
                # 如果schema_value为None或者不是字典类型，使用空字典
                if schema_value is None or not isinstance(schema_value, dict):
                    schema_value = {}
                    
                prompt_info["schema"] = schema_value
                self.prompts_cache.append(prompt_info)
            
            server_info.append(f"{len(self.prompts_cache)} 个提示模板")
        except Exception as e:
            print(f"获取提示列表时出错: {str(e)}")
            self.prompts_cache = []
            server_info.append("提示模板不可用")
        
        print(f"\n已连接到服务器 '{self.name}'，发现 {', '.join(server_info)}")
        self.connected = True
        
    async def call_tool(self, tool_name: str, params: Dict = None) -> Any:
        """调用指定的工具
        
        Args:
            tool_name: 工具名称
            params: 工具参数
            
        Returns:
            工具调用结果
        """
        if not self.session:
            raise ValueError(f"服务器 '{self.name}' 未连接")
            
        if params is None:
            params = {}
        
        try:    
            result = await self.session.call_tool(tool_name, params)
            
            # 确保返回值是可序列化的
            content = result.content
            
            # 检查是否有特殊对象类型需要处理
            if hasattr(content, 'content'):
                return content.content
            elif hasattr(content, 'text'):
                return content.text
            elif hasattr(content, '__str__'):
                return str(content)
            
            return content
        except Exception as e:
            raise ValueError(f"调用工具 '{tool_name}' 失败: {str(e)}")

    async def get_resource(self, resource_uri: str) -> Any:
        """获取指定的资源
        
        Args:
            resource_uri: 资源URI
            
        Returns:
            资源内容
        """
        if not self.session:
            raise ValueError(f"服务器 '{self.name}' 未连接")
            
        result = await self.session.get_resource(resource_uri)
        return result.content

    async def call_prompt(self, prompt_name: str, params: Dict = None) -> str:
        """调用指定的提示模板
        
        Args:
            prompt_name: 提示模板名称
            params: 模板参数
            
        Returns:
            生成的提示文本
        """
        if not self.session:
            raise ValueError(f"服务器 '{self.name}' 未连接")
            
        if params is None:
            params = {}
            
        try:
            # 尝试使用标准的call_prompt方法
            if hasattr(self.session, 'call_prompt') and callable(self.session.call_prompt):
                result = await self.session.call_prompt(prompt_name, params)
                return result.content
            else:
                # 后备：尝试使用call_tool方法调用提示
                print(f"警告: 服务器 '{self.name}' 不支持call_prompt方法，尝试使用call_tool")
                result = await self.session.call_tool(f"prompt_{prompt_name}", params)
                if hasattr(result, 'content'):
                    return result.content
                else:
                    return str(result)
        except Exception as e:
            raise ValueError(f"调用提示模板 '{prompt_name}' 失败: {str(e)}")
        
    async def cleanup(self):
        """清理服务器连接和相关资源"""
        try:
            print(f"开始清理服务器 '{self.name}' 的连接...")
            
            # 首先将连接标记为断开，确保其他操作知道该连接已不可用
            self.connected = False
            
            # 1. 捕获并清理子进程（最优先，避免孤立进程）
            if self.subprocess:
                try:
                    print(f"正在终止子进程: {self.subprocess.pid}")
                    if sys.platform == 'win32':
                        # Windows系统下使用taskkill，但避免等待其完成
                        import subprocess
                        try:
                            # 使用单独的进程执行taskkill，避免阻塞当前进程
                            subprocess.Popen(
                                ['taskkill', '/F', '/T', '/PID', str(self.subprocess.pid)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW
                            )
                        except Exception as e:
                            print(f"启动taskkill进程失败: {e}")
                    else:
                        # Unix平台直接发送SIGKILL信号
                        try:
                            import signal
                            os.kill(self.subprocess.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # 进程已不存在
                        except Exception as e:
                            print(f"发送SIGKILL失败: {e}")
                except Exception as e:
                    print(f"终止子进程时出错: {str(e)}")
                
                # 无需等待子进程结束，直接重置引用
                self.subprocess = None
            
            # 2. 取消所有后台任务（使用更安全的方式）
            remaining_tasks = set()
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    remaining_tasks.add(task)
            
            # 只给任务取消一个非常短的时间
            if remaining_tasks:
                try:
                    # 极短的超时，如果任务无法快速取消，直接放弃等待
                    done, pending = await asyncio.wait(
                        remaining_tasks, 
                        timeout=0.3, 
                        return_when=asyncio.ALL_COMPLETED
                    )
                    if pending:
                        print(f"有 {len(pending)} 个任务未能及时取消，但将继续清理")
                except Exception as e:
                    print(f"等待任务取消时出错: {str(e)}")
            
            self.background_tasks.clear()
            
            # 3. 尝试关闭会话，但不等待它完成
            if self.session:
                # 重置会话状态，优先切断引用
                session = self.session
                self.session = None
                self.stdio = None
                self.write = None
                
                # 然后尝试关闭，但不要等待太久
                if hasattr(session, 'close') and callable(session.close):
                    try:
                        # 创建关闭任务，但设置很短的超时
                        await asyncio.wait_for(session.close(), timeout=0.5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        print(f"关闭会话超时")
                    except Exception as e:
                        print(f"关闭会话时出错: {str(e)}")
            
            # 4. 重新创建exit_stack而不是尝试关闭现有的
            # 这样可以避免遇到"Attempted to exit cancel scope in a different task"错误
            try:
                # 保存引用以便重置
                old_stack = self.exit_stack
                # 立即创建新的exit_stack并重置引用
                self.exit_stack = AsyncExitStack()
                
                # 尝试关闭旧的exit_stack，但不要等待太久
                try:
                    await asyncio.wait_for(old_stack.aclose(), timeout=0.5)
                except (asyncio.TimeoutError, asyncio.CancelledError, RuntimeError):
                    # 忽略所有异常，包括"不同任务"的RuntimeError
                    pass
                except Exception as e:
                    print(f"关闭exit_stack时出错 (已忽略): {str(e)}")
            except Exception:
                # 确保即使出现任何问题，也会重置exit_stack
                self.exit_stack = AsyncExitStack()
                
            print(f"已断开与服务器 '{self.name}' 的连接")
            return True
        except Exception as e:
            print(f"清理服务器 '{self.name}' 连接时出错: {str(e)}")
            # 确保连接标记为断开
            self.connected = False
            # 确保重置关键属性
            self.session = None
            self.stdio = None
            self.write = None
            self.subprocess = None
            self.exit_stack = AsyncExitStack()
            self.background_tasks.clear()
            
            import traceback
            traceback.print_exc()
            return False


class GenericMCPClient:
    """通用型MCP客户端，可以连接到任意MCP服务器"""
    
    def __init__(self):
        """初始化lite_mcp_client"""
        self.connections = {}  # 存储多个连接，键为服务器名称
        self.current_connection = None  # 当前活动连接
        self.server_config = None
        
        # 尝试初始化LLM (如果有API密钥)
        api_base = os.environ.get("OPENAI_API_BASE")
        model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo-16k")
        api_key = os.environ.get("OPENAI_API_KEY")
        provider = os.environ.get("PROVIDER", "openai")
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        anthropic_api_base = os.environ.get("ANTHROPIC_API_BASE")
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
        print(f"provider: {provider}")
        if "openai" in provider and api_key:
            self.llm = ChatOpenAI(
                base_url=api_base, 
                api_key=api_key, 
                model=model_name,
                max_tokens=16000
            )
        elif "google" in provider and google_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                api_key=google_api_key,
                max_tokens=128000,
                timeout=None,
                max_retries=2,
            )
        elif "anthropic" in provider and anthropic_api_key:
            self.llm = ChatAnthropic(
                base_url=anthropic_api_base,
                api_key=anthropic_api_key,
                model=model_name,
                max_tokens_to_sample=128000,
                max_retries=2,
            )
        elif "aws" in provider and aws_access_key_id and aws_secret_access_key:
            self.llm = ChatBedrockConverse(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name="us-east-1",
                model=model_name,
                max_tokens=128000,
                timeout=None,
                temperature=1.0,
                max_retries=2,
            )
            print("警告: 未找到有效的API密钥，智能查询功能将不可用")
            print("请在.env文件中设置OPENAI_API_KEY或GOOGLE_API_KEY")
            self.llm = None

        # 添加对话历史记录
        self.history = []

    def load_server_config(self, config_path: str = DEFAULT_CONFIG_PATH) -> Dict:
        """从配置文件加载服务器配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置的服务器字典
        """
        # 如果配置文件不存在，创建默认配置
        if not os.path.exists(config_path):
            self._create_default_config(config_path)
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if not config.get('mcp_servers'):
                raise ValueError("配置文件中未找到'mcp_servers'部分")
                
            return config
        except json.JSONDecodeError:
            raise ValueError(f"配置文件 {config_path} 不是有效的JSON格式")
        except Exception as e:
            raise ValueError(f"加载配置文件时出错: {str(e)}")
            
    def _create_default_config(self, config_path: str):
        """创建默认配置文件
        
        Args:
            config_path: 配置文件路径
        """
        # 获取脚本当前目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        default_config = {
            "mcp_servers": [
                {
                    "name": "新闻服务",
                    "type": "stdio",
                    "command": "python",
                    "args": [os.path.join(script_dir, "server.py")],
                    "env": {},
                    "description": "新闻聚合服务"
                }
            ],
            "default_server": "新闻服务"
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
            
        print(f"已创建默认配置文件: {config_path}")
    
    def list_available_servers(self, config: Dict) -> None:
        """列出配置中的所有可用服务器
        
        Args:
            config: 配置字典
        """
        servers = config.get('mcp_servers', [])
        if not servers:
            print("没有配置服务器")
            return
            
        print("\n可用服务器:")
        for i, server in enumerate(servers, 1):
            name = server.get('name', f'服务器 {i}')
            server_type = server.get('type', 'unknown')
            desc = server.get('description', '无描述')
            
            if server_type == 'stdio':
                command = server.get('command', '')
                args = ' '.join(server.get('args', []))
                print(f"{i}. {name} [{server_type}] - {desc}")
                print(f"   命令: {command} {args}")
            elif server_type == 'sse':
                url = server.get('url', '')
                print(f"{i}. {name} [{server_type}] - {desc}")
                print(f"   URL: {url}")
            else:
                print(f"{i}. {name} [未知类型: {server_type}] - {desc}")
                
        # 显示默认服务器
        default = config.get('default_server')
        if default:
            if isinstance(default, list):
                print(f"\n默认服务器: {', '.join(default)}")
            else:
                print(f"\n默认服务器: {default}")

    async def connect_to_server_by_name(self, server_name: str, config: Dict) -> bool:
        """通过名称连接到服务器
        
        Args:
            server_name: 服务器名称
            config: 配置字典
            
        Returns:
            连接是否成功
        """
        # 查找服务器配置
        server = None
        for srv in config.get('mcp_servers', []):
            if srv.get('name') == server_name:
                server = srv
                break
            
        if not server:
            print(f"找不到服务器 '{server_name}' 的配置")
            return False
        
        # 检查是否已经连接该服务器，如果是则直接切换
        if server_name in self.connections and self.connections[server_name].connected:
            print(f"已连接到服务器 '{server_name}'，正在切换...")
            self.current_connection = self.connections[server_name]
            return True
        
        # 如果已有连接但标记为断开，先清理它
        if server_name in self.connections and not self.connections[server_name].connected:
            print(f"发现未连接的服务器 '{server_name}'，将重新连接...")
            del self.connections[server_name]
        
        # 创建并连接到服务器
        print(f"正在连接到服务器 '{server_name}'...")
        return await self.connect_to_server_config(server)
    
    async def connect_to_server_config(self, server_config: Dict) -> bool:
        """根据配置连接到服务器
        
        Args:
            server_config: 服务器配置字典
            
        Returns:
            连接是否成功
        """
        server_name = server_config.get('name')
        if not server_name:
            raise ValueError("服务器配置中缺少'name'字段")
        
        # 检查是否已经连接到此服务器
        if server_name in self.connections and self.connections[server_name].connected:
            print(f"已经连接到服务器 '{server_name}'")
            self.current_connection = self.connections[server_name]
            self.server_config = server_config
            return True
        
        # 如果已有同名但未连接的服务器，先移除它
        if server_name in self.connections:
            print(f"发现同名但未连接的服务器 '{server_name}'，将先移除它")
            del self.connections[server_name]
            
        # 创建新连接 - 改用try/except确保异常被捕获
        try:
            # 创建连接对象
            connection = ServerConnection(server_config)
            
            # 尝试连接
            connect_success = await connection.connect()
            if connect_success:
                # 连接成功后才添加到连接列表中，并设为当前连接
                self.connections[server_name] = connection
                self.current_connection = connection
                self.server_config = server_config
                return True
            else:
                # 连接失败，确保资源被清理
                await connection.cleanup()
                print(f"无法连接到服务器 '{server_name}'")
                return False
        except Exception as e:
            print(f"连接到服务器 '{server_name}' 时发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def disconnect_server(self, server_name: str) -> bool:
        """断开与指定服务器的连接
        
        Args:
            server_name: 服务器名称
            
        Returns:
            是否成功断开连接
        """
        if server_name not in self.connections:
            print(f"服务器 '{server_name}' 未连接")
            return False
            
        connection = self.connections[server_name]
        
        # 如果这是当前连接，需要重置当前连接
        if self.current_connection is connection:
            self.current_connection = None
            
        # 先从字典中移除连接，再清理资源
        # 这样即使清理过程中出错，也不会再次尝试清理同一连接
        del self.connections[server_name]
        
        try:
            # 添加超时限制，防止无限期挂起
            result = await asyncio.wait_for(connection.cleanup(), timeout=2.0)
            # 等待短暂时间，确保任何后台任务或子进程都有机会终止
            await asyncio.sleep(0.1)
            return True
        except asyncio.TimeoutError:
            print(f"断开服务器 '{server_name}' 连接超时，但连接已标记为断开")
            # 即使超时，连接清理函数也应该将内部状态设置为断开
            return True
        except asyncio.CancelledError:
            print(f"断开服务器 '{server_name}' 操作被取消")
            return False
        except Exception as e:
            print(f"断开服务器 '{server_name}' 连接时出错: {str(e)}")
            return True  # 即使有错误，连接也已经从列表中移除，可以认为断开成功
        
    async def switch_connection(self, server_name: str) -> bool:
        """切换到另一个已连接的服务器
        
        Args:
            server_name: 要切换到的服务器名称
            
        Returns:
            切换是否成功
        """
        if server_name not in self.connections:
            print(f"未连接到服务器 '{server_name}'")
            return False
            
        connection = self.connections[server_name]
        if not connection.connected:
            print(f"服务器 '{server_name}' 连接已断开")
            return False
            
        self.current_connection = connection
        self.server_config = connection.config
        print(f"已切换到服务器 '{server_name}'")
        return True
            
    async def refresh_server_info(self):
        """刷新当前服务器提供的工具、资源和提示信息"""
        if not self.current_connection:
            raise ValueError("未连接到任何服务器")
            
        await self.current_connection.refresh_server_info()

    async def call_tool(self, tool_name: str, params: Dict = None) -> Any:
        """调用当前连接服务器上的指定工具
        
        Args:
            tool_name: 工具名称
            params: 工具参数
            
        Returns:
            工具调用结果
        """
        if not self.current_connection:
            raise ValueError("未连接到任何服务器")
            
        return await self.current_connection.call_tool(tool_name, params)

    async def get_resource(self, resource_uri: str) -> Any:
        """获取当前连接服务器上的指定资源
        
        Args:
            resource_uri: 资源URI
            
        Returns:
            资源内容
        """
        if not self.current_connection:
            raise ValueError("未连接到任何服务器")
            
        return await self.current_connection.get_resource(resource_uri)

    async def call_prompt(self, prompt_name: str, params: Dict = None) -> str:
        """调用当前连接服务器上的指定提示模板
        
        Args:
            prompt_name: 提示模板名称
            params: 模板参数
            
        Returns:
            生成的提示文本
        """
        if not self.current_connection:
            raise ValueError("未连接到任何服务器")
            
        return await self.current_connection.call_prompt(prompt_name, params)

    async def process_with_llm(self, prompt: str, data: Any) -> str:
        """使用LLM处理数据
        
        Args:
            prompt: 提示文本
            data: 要处理的数据
            
        Returns:
            LLM处理结果
        """
        if not self.llm:
            return "未配置OpenAI API密钥，无法使用LLM功能。"
            
        try:
            # 将数据转换为字符串
            if isinstance(data, dict) or isinstance(data, list):
                data_str = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                data_str = str(data)
                
            # 调用LLM处理
            messages = [HumanMessage(content=f"{prompt}\n\n以下是需要处理的数据:\n{data_str}")]
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            return f"调用LLM时发生错误: {str(e)}"
            
    async def smart_query(self, query: str) -> str:
        """智能查询功能：自动使用LLM确定需要调用的工具并执行查询
        
        Args:
            query: 用户的自然语言查询，如"查询微博热点新闻并总结"
            
        Returns:
            查询结果的总结
        """
        if not self.llm:
            return "未配置OpenAI API密钥，无法使用智能查询功能。"
            
        if not self.connections:
            return "未连接到任何服务器，请先连接到MCP服务器。"
            
        # 准备系统提示，包含可用工具的信息
        tools_info = self._format_tools_for_llm()
        prompts_info = self._format_prompts_for_llm()
        
        print(f"tools_info: {tools_info}")
        # 检查是否是新对话
        new_conversation = not self.history
        
        # 如果是新对话，初始化历史记录
        if new_conversation:
            # 使用三重引号原始字符串，避免f-string中的花括号解析问题
            system_prompt = f"""你是一个智能助手，
可以使用以下工具来完成任务:

{tools_info}

你还可以使用以下提示模板:

{prompts_info}

执行任务时，你可以按照以下步骤进行思考和行动：

1. 分析用户请求，确定需要执行的操作
2. 决定是否需要调用工具，如果需要，决定调用哪个工具以及参数
3. 分析工具返回的结果，决定下一步操作（可能是调用另一个工具，或者直接回答）
4. 当你获得了足够的信息后，为用户提供最终答案

在每次需要调用工具时，请使用以下格式，确保每个ACTION之间有明确的边界：

ACTION: 工具调用
{{
  "tool": "<工具名称>",
  "params": {{
    "<参数名>": "<参数值>"
  }}
}}

注意：一次只能执行一个工具调用，不要在一个响应中包含多个ACTION指令。每次工具调用后，等待结果再决定下一步。

如果你要调用提示模板，使用以下格式：

ACTION: 提示调用
{{
  "prompt": "<提示模板名称>",
  "params": {{
    "<参数名>": "<参数值>"
  }}
}}

当你完成任务并准备给出最终答案时，使用以下格式：

ACTION: 最终回答
<给用户的完整回答>

每次调用工具或提示后，我会将结果提供给你，你需要根据这些结果决定下一步操作。
"""
            # 创建并保存初始的对话历史
            self.history = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        else:
            # 将新的用户查询添加到现有历史记录中
            self.history.append(HumanMessage(content=query))
        
        # 使用历史记录而不是创建新的messages列表
        messages = self.history
        
        try:
            # 初始化循环计数器，防止无限循环
            max_iterations = 20
            iterations = 0
            
            while iterations < max_iterations:
                iterations += 1
                print(f"\n[执行第 {iterations} 次迭代]")
                
                # 调用LLM获取下一步操作
                response = await self.llm.ainvoke(messages)
                llm_content = response.content
                
                # 添加LLM响应到消息历史
                messages.append(AIMessage(content=llm_content))
                
                print(f"\nLLM响应:\n{llm_content}")
                
                # 检查LLM是否决定执行操作
                action_match = re.search(r'ACTION:\s*(.*?)(?:\n|$)', llm_content, re.IGNORECASE)
                if not action_match:
                    # 没有找到ACTION标记，将整个响应作为最终答案
                    return llm_content
                
                action_type = action_match.group(1).strip().lower()
                
                if "最终回答" in action_type:
                    # 直接返回完整的LLM响应，而不是仅提取最终回答部分
                    # 这样HTML代码和其他格式化内容都会被保留
                    return llm_content
                
                elif "工具调用" in action_type:
                    # 提取工具调用JSON - 改进的提取逻辑
                    try:
                        # 获取ACTION后的内容
                        action_pos = llm_content.find('ACTION: 工具调用')
                        if action_pos == -1:
                            action_pos = llm_content.lower().find('action: 工具调用')
                        
                        if action_pos >= 0:
                            # 提取ACTION后面的内容
                            content_after_action = llm_content[action_pos + len('ACTION: 工具调用'):].strip()
                            
                            # 寻找JSON对象
                            json_start = content_after_action.find('{')
                            if json_start >= 0:
                                # 找到匹配的闭合括号
                                bracket_count = 0
                                json_end = -1
                                
                                for i, char in enumerate(content_after_action[json_start:]):
                                    if char == '{':
                                        bracket_count += 1
                                    elif char == '}':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            json_end = json_start + i + 1
                                            break
                                
                                if json_end > 0:
                                    json_str = content_after_action[json_start:json_end]
                                    # 清理JSON字符串中可能存在的Markdown格式
                                    json_str = re.sub(r'```json|```', '', json_str).strip()
                                    
                                    tool_call = json.loads(json_str)
                                    
                                    tool_spec = tool_call.get("tool")
                                    params = tool_call.get("params", {})
                                    
                                    if not tool_spec:
                                        # 工具名称为空，添加错误消息并继续
                                        error_msg = "工具调用中缺少工具名称。格式应为: {\"tool\": \"服务器名.工具名\", \"params\": {...}}"
                                        messages.append(HumanMessage(content=error_msg))
                                        continue
                                    
                                    # 解析服务器名和工具名
                                    if '.' in tool_spec:
                                        server_name, tool_name = tool_spec.split('.', 1)
                                    else:
                                        # 如果没有指定服务器，使用第一个连接的服务器
                                        if not self.connections:
                                            error_msg = "没有指定服务器名，并且没有连接到任何服务器。"
                                            messages.append(HumanMessage(content=error_msg))
                                            continue
                                            
                                        server_name = next(iter(self.connections.keys()))
                                        tool_name = tool_spec
                                        messages.append(HumanMessage(content=f"你没有指定服务器名，我将使用默认服务器 '{server_name}'。"))
                                    
                                    print(f"\n正在调用服务器 '{server_name}' 上的工具: {tool_name}")
                                    print(f"参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
                                    
                                    try:
                                        # 执行工具调用
                                        result = await self.call_tool_by_server(server_name, tool_name, params)
                                        
                                        # 将结果反馈给LLM - 改进的序列化处理
                                        result_msg = f"工具 {server_name}.{tool_name} 的调用结果:\n\n"
                                        
                                        if result is None:
                                            result_msg += "无结果数据"
                                        elif isinstance(result, (dict, list)):
                                            try:
                                                result_msg += json.dumps(result, ensure_ascii=False, indent=2)
                                            except TypeError:
                                                # 如果JSON序列化失败，使用字符串表示
                                                result_msg += str(result)
                                        else:
                                            result_msg += str(result)

                                        print(f"\n调用{tool_name}结果:\n{result_msg}")  

                                        messages.append(HumanMessage(content=result_msg))
                                    except Exception as tool_error:
                                        error_msg = f"执行工具 {server_name}.{tool_name} 调用时出错: {str(tool_error)}"
                                        print(error_msg)
                                        messages.append(HumanMessage(content=error_msg))
                                else:
                                    error_msg = "无法找到完整的JSON对象。请确保你的JSON格式正确，并包含完整的花括号。"
                                    messages.append(HumanMessage(content=error_msg))
                            else:
                                error_msg = "在ACTION后找不到JSON对象。请使用正确的格式：ACTION: 工具调用\n{\"tool\": \"服务器名.工具名\", \"params\": {...}}"
                                messages.append(HumanMessage(content=error_msg))
                        else:
                            error_msg = "无法解析工具调用。请使用正确的格式：ACTION: 工具调用\n{\"tool\": \"服务器名.工具名\", \"params\": {...}}"
                            messages.append(HumanMessage(content=error_msg))
                            
                    except json.JSONDecodeError as e:
                        error_msg = f"解析工具调用JSON时出错: {str(e)}\n请提供有效的JSON格式，确保所有引号和逗号都正确放置。例如:\n{{\"tool\": \"服务器名.工具名\", \"params\": {{\"source\": \"weibo\"}}}}"
                        print(error_msg)
                        messages.append(HumanMessage(content=error_msg))
                        
                    except Exception as e:
                        error_msg = f"执行工具调用时出错: {str(e)}"
                        print(error_msg)
                        import traceback
                        traceback.print_exc()
                        messages.append(HumanMessage(content=error_msg))
                
                elif "提示调用" in action_type:
                    # 提取提示调用JSON - 使用相同的改进逻辑
                    try:
                        # 获取ACTION后的内容
                        action_pos = llm_content.find('ACTION: 提示调用')
                        if action_pos == -1:
                            action_pos = llm_content.lower().find('action: 提示调用')
                        
                        if action_pos >= 0:
                            # 提取ACTION后面的内容
                            content_after_action = llm_content[action_pos + len('ACTION: 提示调用'):].strip()
                            
                            # 寻找JSON对象
                            json_start = content_after_action.find('{')
                            if json_start >= 0:
                                # 找到匹配的闭合括号
                                bracket_count = 0
                                json_end = -1
                                
                                for i, char in enumerate(content_after_action[json_start:]):
                                    if char == '{':
                                        bracket_count += 1
                                    elif char == '}':
                                        bracket_count -= 1
                                        if bracket_count == 0:
                                            json_end = json_start + i + 1
                                            break
                                
                                if json_end > 0:
                                    json_str = content_after_action[json_start:json_end]
                                    # 清理JSON字符串中可能存在的Markdown格式
                                    json_str = re.sub(r'```json|```', '', json_str).strip()
                                    
                                    prompt_call = json.loads(json_str)
                                    
                                    prompt_spec = prompt_call.get("prompt")
                                    params = prompt_call.get("params", {})
                                    
                                    if not prompt_spec:
                                        # 提示名称为空，添加错误消息并继续
                                        error_msg = "提示调用中缺少提示模板名称。格式应为: {\"prompt\": \"服务器名.提示名\", \"params\": {...}}"
                                        messages.append(HumanMessage(content=error_msg))
                                        continue
                                    
                                    # 解析服务器名和提示名
                                    if '.' in prompt_spec:
                                        server_name, prompt_name = prompt_spec.split('.', 1)
                                    else:
                                        # 如果没有指定服务器，使用第一个连接的服务器
                                        if not self.connections:
                                            error_msg = "没有指定服务器名，并且没有连接到任何服务器。"
                                            messages.append(HumanMessage(content=error_msg))
                                            continue
                                            
                                        server_name = next(iter(self.connections.keys()))
                                        prompt_name = prompt_spec
                                        messages.append(HumanMessage(content=f"你没有指定服务器名，我将使用默认服务器 '{server_name}'。"))
                                    
                                    print(f"\n正在调用服务器 '{server_name}' 上的提示模板: {prompt_name}")
                                    print(f"参数: {json.dumps(params, ensure_ascii=False, indent=2)}")
                                    
                                    try:
                                    # 执行提示调用
                                        result = await self.call_prompt_by_server(server_name, prompt_name, params)
                                    
                                    # 将结果反馈给LLM
                                        result_msg = f"提示模板 {server_name}.{prompt_name} 的调用结果:\n\n{result}"
                                        messages.append(HumanMessage(content=result_msg))
                                    except Exception as e:
                                        error_msg = f"执行提示调用 {server_name}.{prompt_name} 时出错: {str(e)}"
                                        print(error_msg)
                                        messages.append(HumanMessage(content=error_msg))
                                else:
                                    error_msg = "无法找到完整的JSON对象。请确保你的JSON格式正确，并包含完整的花括号。"
                                    messages.append(HumanMessage(content=error_msg))
                            else:
                                error_msg = "在ACTION后找不到JSON对象。请使用正确的格式：ACTION: 提示调用\n{\"prompt\": \"服务器名.提示名\", \"params\": {...}}"
                                messages.append(HumanMessage(content=error_msg))
                        else:
                            error_msg = "无法解析提示调用。请使用正确的格式：ACTION: 提示调用\n{\"prompt\": \"服务器名.提示名\", \"params\": {...}}"
                            messages.append(HumanMessage(content=error_msg))
                            
                    except json.JSONDecodeError as e:
                        error_msg = f"解析提示调用JSON时出错: {str(e)}\n请提供有效的JSON格式，确保所有引号和逗号都正确放置。"
                        print(error_msg)
                        messages.append(HumanMessage(content=error_msg))
                        
                    except Exception as e:
                        error_msg = f"执行提示调用时出错: {str(e)}"
                        print(error_msg)
                        messages.append(HumanMessage(content=error_msg))
                
                else:
                    # 未识别的操作类型，请求澄清
                    messages.append(HumanMessage(content=f"未识别的操作类型: {action_type}。请使用正确的ACTION格式。"))
            
            # 如果达到最大迭代次数，直接生成最终答案
            messages.append(HumanMessage(content="已达到最大迭代次数。请提供最终答案。"))
            final_response = await self.llm.ainvoke(messages)
            return final_response.content
            
        except Exception as e:
            print(f"\n执行智能查询时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"执行查询时发生错误: {str(e)}"
            
    def _format_tools_for_llm(self) -> str:
        """格式化所有已连接服务器的工具信息，用于LLM提示"""
        if not self.connections:
            return "没有可用的工具。"
            
        tools_text = []
        
        # 遍历所有已连接服务器
        for server_name, connection in self.connections.items():
            if not connection.connected:
                continue
                
            # 添加服务器标题
            tools_text.append(f"## 服务器: {server_name}")
            
            # 添加该服务器的所有工具
            if not connection.tools_cache:
                tools_text.append("此服务器没有可用工具。")
                continue
                    
            for tool in connection.tools_cache:
                name = tool['name']
                desc = tool['description']
                schema = tool['schema']
            
                params_text = ""
                if isinstance(schema, dict) and 'properties' in schema:
                    params_text = "参数:\n"
                    required_params = schema.get('required', [])
                    
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', '任意类型')
                        param_desc = param_info.get('description', '')
                        required = "*" if param_name in required_params else ""
                        params_text += f"  - {param_name}{required}: {param_type} - {param_desc}\n"
            
                tools_text.append(f"工具: {server_name}.{name}\n描述: {desc}\n{params_text}")
        
        return "\n".join(tools_text)
        
    def _format_prompts_for_llm(self) -> str:
        """格式化所有已连接服务器的提示模板信息，用于LLM提示"""
        if not self.connections:
            return "没有可用的提示模板。"
            
        prompts_text = []
        
        # 遍历所有已连接服务器
        for server_name, connection in self.connections.items():
            if not connection.connected:
                continue
                
            # 添加服务器标题
            prompts_text.append(f"## 服务器: {server_name}")
            
            # 添加该服务器的所有提示模板
            if not connection.prompts_cache:
                prompts_text.append("此服务器没有可用提示模板。")
                continue
                
            for prompt in connection.prompts_cache:
                name = prompt['name']
                desc = prompt['description']
                schema = prompt['schema']
            
                params_text = ""
                if isinstance(schema, dict) and 'properties' in schema:
                    params_text = "参数:\n"
                    required_params = schema.get('required', [])
                    
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', '任意类型')
                        param_desc = param_info.get('description', '')
                        required = "*" if param_name in required_params else ""
                        params_text += f"  - {param_name}{required}: {param_type} - {param_desc}\n"
            
                prompts_text.append(f"提示模板: {server_name}.{name}\n描述: {desc}\n{params_text}")
            
        return "\n".join(prompts_text)


    async def call_tool_by_server(self, server_name: str, tool_name: str, params: Dict = None) -> Any:
        """调用指定服务器上的工具
        
        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            params: 工具参数
            
        Returns:
            工具调用结果
        """
        if server_name not in self.connections:
            raise ValueError(f"未连接到服务器 '{server_name}'")
            
        connection = self.connections[server_name]
        if not connection.connected:
            raise ValueError(f"服务器 '{server_name}' 连接已断开")
            
        return await connection.call_tool(tool_name, params)

    async def call_prompt_by_server(self, server_name: str, prompt_name: str, params: Dict = None) -> str:
        """调用指定服务器上的提示模板
        
        Args:
            server_name: 服务器名称
            prompt_name: 提示模板名称
            params: 模板参数
        
        Returns:
            生成的提示文本
        """
        if server_name not in self.connections:
            raise ValueError(f"未连接到服务器 '{server_name}'")
            
        connection = self.connections[server_name]
        if not connection.connected:
            raise ValueError(f"服务器 '{server_name}' 连接已断开")
            
        return await connection.call_prompt(prompt_name, params)

    async def cleanup(self):
        """清理所有资源"""
        # 复制连接名称列表，因为在循环中会修改字典
        server_names = list(self.connections.keys())
        
        for server_name in server_names:
            await self.disconnect_server(server_name)
            
        # 确保连接字典和当前连接被重置
        self.connections = {}
        self.current_connection = None

    async def interactive_mode(self):
        """运行交互式命令行界面"""
        print("\nlite_mcp_client已启动!")
        print("可用命令:")
        print("  connect <服务器名> - 连接到指定服务器")
        print("  connect-all - 连接到所有默认服务器")
        print("  disconnect <服务器名> - 断开与指定服务器的连接")
        print("  switch <服务器名> - 切换到已连接的服务器")
        print("  connections - 列出所有连接")
        print("  tools - 列出所有已连接服务器的工具")
        print("  resources - 列出所有已连接服务器的资源")
        print("  prompts - 列出所有已连接服务器的提示模板")
        print("  call <服务器名>.<工具名> [参数JSON] - 调用指定服务器上的工具")
        print("  get <服务器名>.<资源URI> - 获取指定服务器上的资源")
        print("  prompt <服务器名>.<提示名> [参数JSON] - 使用指定服务器上的提示模板")
        print("  ask <自然语言提问> - 智能处理提问，自动调用所需工具")
        print("  clear-history - 清除对话历史，开始新对话")
        print("  help - 显示帮助信息")
        print("  quit - 退出程序")
        
        # 用于跟踪是否应该退出循环
        should_exit = False
        
        while not should_exit:
            try:
                cmd = input("\n命令: ").strip()
                
                if cmd.lower() == 'quit':
                    should_exit = True
                    import os
                    os._exit(0)

                    
                elif cmd.lower() == 'help':
                    print("\n可用命令:")
                    print("  connect <服务器名> - 连接到指定服务器")
                    print("  connect-all - 连接到所有默认服务器")
                    print("  disconnect <服务器名> - 断开与指定服务器的连接")
                    print("  switch <服务器名> - 切换到已连接的服务器")
                    print("  connections - 列出所有连接")
                    print("  tools - 列出所有已连接服务器的工具")
                    print("  resources - 列出所有已连接服务器的资源")
                    print("  prompts - 列出所有已连接服务器的提示模板")
                    print("  call <服务器名>.<工具名> [参数JSON] - 调用指定服务器上的工具")
                    print("  get <服务器名>.<资源URI> - 获取指定服务器上的资源")
                    print("  prompt <服务器名>.<提示名> [参数JSON] - 使用指定服务器上的提示模板")
                    print("  ask <自然语言提问> - 智能处理提问，自动调用所需工具")
                    print("  clear-history - 清除对话历史，开始新对话")
                    print("  help - 显示帮助信息")
                    print("  quit - 退出程序")
                    
                elif cmd.lower() == 'clear-history':
                    self.history = []
                    print("\n对话历史已清除。开始新对话。")
                    
                elif cmd.lower().startswith('connect '):
                    server_name = cmd[8:].strip()
                    try:
                        # 先保存当前连接，以便连接失败时恢复
                        old_current = self.current_connection
                        
                        config = self.load_server_config()
                        connect_success = await self.connect_to_server_by_name(server_name, config)
                        
                        if connect_success:
                            print(f"成功连接到服务器 '{server_name}'")
                            
                            # 显示服务器信息
                            if self.current_connection and self.current_connection.connected:
                                tools_count = len(self.current_connection.tools_cache)
                                has_resources = "可用" if self.current_connection.resources_cache else "不可用"
                                prompts_count = len(self.current_connection.prompts_cache)
                                print(f"已连接到服务器 '{server_name}'，发现 {tools_count} 个工具, 资源列表{has_resources}, {prompts_count} 个提示模板")
                        else:
                            # 恢复旧的当前连接
                            self.current_connection = old_current
                            print(f"无法连接到服务器 '{server_name}'")
                    except Exception as e:
                        print(f"连接到服务器 '{server_name}' 时出错: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                elif cmd.lower() == 'connect-all':
                    try:
                        config = self.load_server_config()
                        await self.connect_all_default_servers(config)
                    except Exception as e:
                        print(f"连接到默认服务器时出错: {str(e)}")
                
                elif cmd.lower().startswith('disconnect '):
                    server_name = cmd[11:].strip()
                    try:
                        disconnect_success = await self.disconnect_server(server_name)
                        if not disconnect_success:
                            print(f"断开服务器 '{server_name}' 连接失败")
                    except Exception as e:
                        print(f"断开服务器 '{server_name}' 连接时出错: {str(e)}")
                
                elif cmd.lower().startswith('switch '):
                    server_name = cmd[7:].strip()
                    self.switch_to_server(server_name)
                
                elif cmd.lower() == 'connections':
                    self.list_connections()
                
                elif cmd.lower() == 'tools':
                    self.list_tools()
                
                elif cmd.lower() == 'resources':
                    self.list_resources()
                
                elif cmd.lower() == 'prompts':
                    self.list_prompts()
                    
                elif cmd.lower().startswith('call '):
                    parts = cmd[5:].strip().split(maxsplit=1)
                    server_tool = parts[0]
                    params = {}
                    
                    if '.' not in server_tool:
                        print("\n请指定服务器和工具，格式: 服务器名.工具名")
                        continue
                        
                    server_name, tool_name = server_tool.split('.', 1)
                    
                    if len(parts) > 1:
                        try:
                            params = json.loads(parts[1])
                        except json.JSONDecodeError:
                            print(f"\n参数格式错误，请提供有效的JSON。")
                            continue
                    
                    print(f"\n正在调用服务器 '{server_name}' 上的工具: {tool_name}")
                    try:
                        result = await self.call_tool_by_server(server_name, tool_name, params)
                        print(f"\n结果:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
                    except Exception as e:
                        print(f"\n调用失败: {str(e)}")
                    
                elif cmd.lower().startswith('get '):
                    resource_uri = cmd[4:].strip()
                    
                    if '.' in resource_uri:
                        # 格式: get 服务器名.资源URI
                        server_name, resource_uri = resource_uri.split('.', 1)
                        print(f"\n正在获取服务器 '{server_name}' 上的资源: {resource_uri}")
                        try:
                            result = await self.get_resource_by_server(server_name, resource_uri)
                            print(f"\n结果:\n{result}")
                        except Exception as e:
                            print(f"\n获取资源失败: {str(e)}")
                    else:
                        # 兼容旧格式: get 资源URI
                        if not self.current_connection:
                            print("\n未指定服务器，请先切换到一个服务器或使用格式: get 服务器名.资源URI")
                            continue
                            
                        print(f"\n正在获取资源: {resource_uri}")
                        try:
                            result = await self.current_connection.get_resource(resource_uri)
                            print(f"\n结果:\n{result}")
                        except Exception as e:
                            print(f"\n获取资源失败: {str(e)}")
                    
                elif cmd.lower().startswith('prompt '):
                    parts = cmd[7:].strip().split(maxsplit=1)
                    prompt_spec = parts[0]
                    params = {}
                    
                    if '.' in prompt_spec:
                        # 格式: prompt 服务器名.提示名 [参数JSON]
                        server_name, prompt_name = prompt_spec.split('.', 1)
                    
                        if len(parts) > 1:
                            try:
                                params = json.loads(parts[1])
                            except json.JSONDecodeError:
                                print(f"\n参数格式错误，请提供有效的JSON。")
                                continue
                        
                        print(f"\n正在使用服务器 '{server_name}' 上的提示模板: {prompt_name}")
                        try:
                            result = await self.call_prompt_by_server(server_name, prompt_name, params)
                            print(f"\n生成的提示:\n{result}")
                        except Exception as e:
                            print(f"\n调用提示模板失败: {str(e)}")
                    else:
                        # 兼容旧格式
                        if not self.current_connection:
                            print("\n未指定服务器，请先切换到一个服务器或使用格式: prompt 服务器名.提示名")
                            continue
                            
                        prompt_name = prompt_spec
                        params = {}
                        
                        if len(parts) > 1:
                            try:
                                params = json.loads(parts[1])
                            except json.JSONDecodeError:
                                print(f"\n参数格式错误，请提供有效的JSON。")
                                continue
                        
                        print(f"\n正在使用提示模板: {prompt_name}")
                        try:
                            result = await self.current_connection.call_prompt(prompt_name, params)
                            print(f"\n生成的提示:\n{result}")
                        except Exception as e:
                            print(f"\n调用提示模板失败: {str(e)}")
                    
                elif cmd.lower().startswith('ask '):
                    query = cmd[4:].strip()
                    print(f"\n处理查询: {query}")
                    
                    result = await self.smart_query(query)
                    # print(f"\n回答:\n{result}")
                    
                else:
                    print(f"未知命令: {cmd}")
                    print("输入 'help' 查看可用命令")
                    
            except KeyboardInterrupt:
                print("\n检测到Ctrl+C，正在退出...")
                should_exit = True
                break
            except Exception as e:
                print(f"错误: {str(e)}")
                import traceback
                traceback.print_exc()

        print("交互模式已结束")


    async def get_resource_by_server(self, server_name: str, resource_uri: str) -> Any:
        """获取指定服务器上的资源
        
        Args:
            server_name: 服务器名称
            resource_uri: 资源URI
            
        Returns:
            资源内容
        """
        if server_name not in self.connections:
            raise ValueError(f"未连接到服务器 '{server_name}'")
            
        connection = self.connections[server_name]
        if not connection.connected:
            raise ValueError(f"服务器 '{server_name}' 连接已断开")
            
        return await connection.get_resource(resource_uri)

    def list_tools(self):
        """列出所有已连接服务器的工具"""
        if not self.connections:
            print("\n没有连接到任何服务器。")
            return
            
        print("\n所有已连接服务器的可用工具:")
        for server_name, conn in self.connections.items():
            if not conn.connected:
                continue
                
            print(f"\n## 服务器: {server_name}")
            if not conn.tools_cache:
                print("  没有可用的工具。")
                continue
                
            for i, tool in enumerate(conn.tools_cache, 1):
                print(f"\n  {i}. {tool['name']}")
                print(f"     描述: {tool['description']}")
                print(f"     参数:")
                if isinstance(tool['schema'], dict) and 'properties' in tool['schema']:
                    for param_name, param_info in tool['schema']['properties'].items():
                        param_type = param_info.get('type', '任意类型')
                        param_desc = param_info.get('description', '')
                        required = "*" if 'required' in tool['schema'] and param_name in tool['schema']['required'] else ""
                        print(f"       - {param_name}{required}: {param_type} - {param_desc}")
                else:
                    print("       无参数信息")

    def list_resources(self):
        """列出所有已连接服务器的资源"""
        if not self.connections:
            print("\n没有连接到任何服务器。")
            return
            
        print("\n所有已连接服务器的可用资源:")
        for server_name, conn in self.connections.items():
            if not conn.connected:
                continue
                
            print(f"\n## 服务器: {server_name}")
            if not conn.resources_cache:
                print("  没有可用的资源。")
                continue
                
            for i, resource in enumerate(conn.resources_cache, 1):
                print(f"  {i}. {resource['uri']}")
                if resource['description']:
                    print(f"     描述: {resource['description']}")

    def list_prompts(self):
        """列出所有已连接服务器的提示模板"""
        if not self.connections:
            print("\n没有连接到任何服务器。")
            return
            
        print("\n所有已连接服务器的可用提示模板:")
        for server_name, conn in self.connections.items():
            if not conn.connected:
                continue
                
            print(f"\n## 服务器: {server_name}")
            if not conn.prompts_cache:
                print("  没有可用的提示模板。")
                continue
                
            for i, prompt in enumerate(conn.prompts_cache, 1):
                print(f"\n  {i}. {prompt['name']}")
                print(f"     描述: {prompt['description']}")
                print(f"     参数:")
                if isinstance(prompt['schema'], dict) and 'properties' in prompt['schema']:
                    for param_name, param_info in prompt['schema']['properties'].items():
                        param_type = param_info.get('type', '任意类型')
                        param_desc = param_info.get('description', '')
                        required = "*" if 'required' in prompt['schema'] and param_name in prompt['schema']['required'] else ""
                        print(f"       - {param_name}{required}: {param_type} - {param_desc}")
                else:
                    print("       无参数信息")

    def list_connections(self):
        """列出所有当前的连接状态"""
        if not self.connections:
            print("\n没有连接到任何服务器。")
            return
            
        print("\n当前连接:")
        for server_name, conn in self.connections.items():
            status = "已连接" if conn.connected else "已断开"
            current = " (当前活动)" if conn is self.current_connection else ""
            print(f"  {server_name}: {status}{current}")

    def switch_to_server(self, server_name: str) -> bool:
        """切换到指定的已连接服务器
        
        Args:
            server_name: 服务器名称
            
        Returns:
            切换是否成功
        """
        if server_name not in self.connections:
            print(f"未连接到服务器 '{server_name}'")
            return False
            
        connection = self.connections[server_name]
        if not connection.connected:
            print(f"服务器 '{server_name}' 连接已断开")
            return False
            
        self.current_connection = connection
        self.server_config = connection.config
        print(f"已切换到服务器 '{server_name}'")
        return True
    
    async def connect_all_default_servers(self, config: Dict) -> bool:
        """连接到所有默认服务器
        
        Args:
            config: 配置字典
            
        Returns:
            是否至少成功连接一个服务器
        """
        connected = False
        
        # 获取默认服务器列表 - 支持顶级default_server配置
        default_server_names = config.get('default_server', [])
        
        # 处理可能的字符串情况
        if isinstance(default_server_names, str):
            default_server_names = [default_server_names]
        
        if not default_server_names:
            print("配置中没有默认服务器")
            return False
            
        # 逐个连接默认服务器
        for server_name in default_server_names:
            print(f"正在连接到默认服务器: {server_name}")
            
            try:
                # 查找服务器配置
                server_config = None
                for server in config.get('mcp_servers', []):
                    if server.get('name') == server_name:
                        server_config = server
                        break
                        
                if not server_config:
                    print(f"找不到服务器 '{server_name}' 的配置")
                    continue
                    
                # 尝试连接，但捕获所有异常
                success = await self.connect_to_server_config(server_config)
                
                if success:
                    connected = True
                    print(f"成功连接到默认服务器: {server_name}")
                else:
                    print(f"无法连接到默认服务器: {server_name}")
                    
            except Exception as e:
                print(f"连接到默认服务器 '{server_name}' 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if not connected:
            print("无法连接到任何默认服务器")
        
        return connected
    
    async def server_management_mode(self, config_path: str) -> bool:
        """服务器管理模式，用于管理服务器连接
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            是否成功连接到服务器
        """
        config = self.load_server_config(config_path)
        self.list_available_servers(config)
        
        print("\n服务器管理模式")
        print("可用命令:")
        print("  connect <服务器名/编号> - 连接到指定服务器")
        print("  connect-all - 连接到所有默认服务器")
        print("  list - 列出所有可用服务器")
        print("  quit - 退出管理模式")
        
        while True:
            cmd = input("\n管理命令: ").strip()
            
            if cmd.lower() == 'quit':
                return False
                
            elif cmd.lower() == 'list':
                self.list_available_servers(config)
                
            elif cmd.lower() == 'connect-all':
                if await self.connect_all_default_servers(config):
                    return True
                    
            elif cmd.lower().startswith('connect '):
                server_spec = cmd[8:].strip()
                
                # 检查是否是数字(服务器编号)
                try:
                    if server_spec.isdigit():
                        idx = int(server_spec) - 1
                        if 0 <= idx < len(config.get('mcp_servers', [])):
                            server_name = config['mcp_servers'][idx].get('name')
                            if await self.connect_to_server_by_name(server_name, config):
                                return True
                        else:
                            print(f"无效的服务器编号: {server_spec}")
                    else:
                        if await self.connect_to_server_by_name(server_spec, config):
                            return True
                except Exception as e:
                    print(f"连接到服务器时出错: {str(e)}")
            else:
                print(f"未知命令: {cmd}")
        
            return False

async def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MCP 客户端')
    parser.add_argument('--server', help='要连接的服务器名称')
    parser.add_argument('--connect-all', action='store_true', help='连接到所有默认服务器')
    parser.add_argument('--query', help='要执行的查询')
    parser.add_argument('--call', help='要调用的工具，格式: 服务器名.工具名')
    parser.add_argument('--params', help='工具参数 (JSON 格式)')
    parser.add_argument('--get', help='要获取的资源，格式: 服务器名.资源URI')
    parser.add_argument('--prompt', help='要使用的提示模板，格式: 服务器名.提示名')
    parser.add_argument('--interactive', action='store_true', help='启动交互式模式')
    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG_PATH, help='配置文件路径')
    parser.add_argument('direct_query', nargs='?', help='直接查询（无需--query前缀）')
    args = parser.parse_args()

    # 创建客户端实例
    client = GenericMCPClient()
    
    try:
        # 加载配置
        config_path = args.config if args.config else None
        config = client.load_server_config(config_path)
        
        # 处理连接
        if args.server:
            await client.connect_to_server_by_name(args.server, config)
        elif args.connect_all or args.direct_query or not any([args.query, args.call, args.get, args.prompt, args.interactive]):
            # 如果没有指定服务器，或者有直接查询，默认连接所有服务器
            await client.connect_all_default_servers(config)
        
        # 处理命令行操作
        if args.query:
            result = await client.smart_query(args.query)
            with open('result.txt', 'w', encoding='utf-8') as f:
                f.write(result)
            print(result)
        elif args.direct_query:
            # 处理直接查询参数
            result = await client.smart_query(args.direct_query)
            with open('result.txt', 'w', encoding='utf-8') as f:
                f.write(result)
            print(result)
        elif args.call:
            if '.' not in args.call:
                print("错误: --call 参数必须使用格式 '服务器名.工具名'")
                return
                
            server_name, tool_name = args.call.split('.', 1)
            params = {}
            if args.params:
                try:
                    params = json.loads(args.params)
                except json.JSONDecodeError:
                    print("错误: --params 必须是有效的 JSON")
                    return
                    
            result = await client.call_tool_by_server(server_name, tool_name, params)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.get:
            if '.' not in args.get:
                print("错误: --get 参数必须使用格式 '服务器名.资源URI'")
                return
                
            server_name, resource_uri = args.get.split('.', 1)
            result = await client.get_resource_by_server(server_name, resource_uri)
            print(result)
        elif args.prompt:
            if '.' not in args.prompt:
                print("错误: --prompt 参数必须使用格式 '服务器名.提示名'")
                return
                
            server_name, prompt_name = args.prompt.split('.', 1)
            params = {}
            if args.params:
                try:
                    params = json.loads(args.params)
                except json.JSONDecodeError:
                    print("错误: --params 必须是有效的 JSON")
                    return
                    
            result = await client.call_prompt_by_server(server_name, prompt_name, params)
            print(result)
        elif args.interactive or not any([args.query, args.call, args.get, args.prompt, args.direct_query]):
            # 如果没有指定其他操作，或者明确要求交互模式，则启动交互模式
            await client.interactive_mode()
    finally:
        # 确保连接被清理
        print("正在清理连接和资源...")
        try:
            # 仅给清理程序较短的时间，避免卡住
            cleanup_task = asyncio.create_task(client.cleanup())
            await asyncio.wait_for(cleanup_task, timeout=3.0)
        except asyncio.TimeoutError:
            print("清理连接超时，忽略并继续执行")
        except Exception as e:
            print(f"清理连接时出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 立即取消所有任务而不是等待它们
        print("正在终止所有剩余任务...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        # 非常短的等待时间
        if tasks:
            try:
                await asyncio.wait(tasks, timeout=0.5)
                print("所有任务已取消")
            except Exception:
                print("任务取消中出现错误，强制退出")
                
        # 在某些情况下可能需要强制退出
        print("程序终止")
        import os, sys
        # 使用os._exit强制退出，确保不会卡住
        os._exit(0)



if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("程序已正常退出")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序终止")
