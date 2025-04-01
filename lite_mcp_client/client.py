"""通用型MCP客户端核心"""

import asyncio
import os
import json
from typing import Dict, List, Any, Optional, Set, Tuple

from dotenv import load_dotenv

from .connection import ServerConnection
from .config import load_server_config, list_available_servers
from .llm_processing import LLMProcessor
from .utils import parse_json_params, parse_server_and_name, print_help_message

# 尝试导入LLM库
try:
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_anthropic import ChatAnthropic
    from langchain_aws import ChatBedrockConverse
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class GenericMCPClient:
    """通用型MCP客户端，可以连接到任意MCP服务器"""

    def __init__(self):
        """初始化lite_mcp_client"""
        self.connections: Dict[str, ServerConnection] = {}  # 存储多个连接，键为服务器名称
        self.current_connection: Optional[ServerConnection] = None  # 当前活动连接
        self.server_config: Optional[Dict] = None # Store config of current connection

        # 加载环境变量
        load_dotenv()  # 加载.env文件中的环境变量

        # 初始化LLM
        self.llm = self._initialize_llm()
        
        # 初始化LLM处理器
        self.llm_processor = LLMProcessor(self)

    def _initialize_llm(self):
        """初始化LLM（语言模型）
        
        Returns:
            初始化的LLM实例，如果初始化失败则返回None
        """
        if not LANGCHAIN_AVAILABLE:
            print("\n警告: 缺少必要的 LangChain 库。LLM回答 ('ask') 功能不可用。")
            print("请安装必要的包: 'pip install langchain-openai langchain-google-genai langchain-anthropic langchain-aws'")
            return None
        
        try:
            provider = os.environ.get("PROVIDER", "openai").lower()
            api_key = os.environ.get("OPENAI_API_KEY")
            api_base = os.environ.get("OPENAI_API_BASE") # Optional
            model_name = os.environ.get("MODEL_NAME") # Optional, sensible defaults below

            google_api_key = os.environ.get("GOOGLE_API_KEY")
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
            anthropic_api_base = os.environ.get("ANTHROPIC_API_BASE") # Optional
            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            aws_region = os.environ.get("AWS_REGION")
            
            # 默认不使用流式响应
            streaming = False
            
            # 根据提供商初始化对应的LLM
            if provider == "openai":
                if not api_key:
                    print("警告: OPENAI_API_KEY 未设置，LLM功能不可用。")
                    return None
                
                # 设置可选参数
                kwargs = {}
                if api_base:
                    kwargs["openai_api_base"] = api_base
                if not model_name:
                    model_name = "gpt-3.5-turbo" # 默认模型
                    
                return ChatOpenAI(
                    openai_api_key=api_key,
                    model_name=model_name,
                    streaming=streaming,
                    **kwargs
                )
                
            elif provider == "google":
                if not google_api_key:
                    print("警告: GOOGLE_API_KEY 未设置，LLM功能不可用。")
                    return None
                
                if not model_name:
                    model_name = "gemini-pro" # 默认模型
                    
                return ChatGoogleGenerativeAI(
                    google_api_key=google_api_key,
                    model=model_name,
                    streaming=streaming,
                )
                
            elif provider == "anthropic":
                if not anthropic_api_key:
                    print("警告: ANTHROPIC_API_KEY 未设置，LLM功能不可用。")
                    return None
                
                # 设置可选参数
                kwargs = {}
                if anthropic_api_base:
                    kwargs["anthropic_api_url"] = anthropic_api_base
                if not model_name:
                    model_name = "claude-3-haiku-20240307" # 默认模型
                    
                return ChatAnthropic(
                    anthropic_api_key=anthropic_api_key,
                    model_name=model_name,
                    streaming=streaming,
                    **kwargs
                )
                
            elif provider == "aws":
                if not aws_access_key or not aws_secret_key or not aws_region:
                    print("警告: AWS凭证未完全设置，LLM功能不可用。")
                    return None
                
                if not model_name:
                    model_name = "anthropic.claude-3-sonnet-20240229-v1:0" # 默认模型
                    
                return ChatBedrockConverse(
                    model_id=model_name,
                    region_name=aws_region,
                    streaming=streaming,
                )
                
            else:
                print(f"警告: 不支持的LLM提供商: {provider}。支持的选项有: openai, google, anthropic, aws")
                return None
                
        except Exception as e:
            print(f"初始化LLM时出错: {str(e)}")
            return None

    def load_server_config(self, config_path: str) -> Dict:
        """加载并验证服务器配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            加载的配置字典
        """
        self.server_config = load_server_config(config_path)
        return self.server_config

    async def connect_to_server_by_name(self, server_name: str, config: Optional[Dict] = None) -> bool:
        """根据名称连接到指定的服务器
        
        Args:
            server_name: 服务器名称
            config: 可选的配置字典，如果未提供则使用已加载的配置
            
        Returns:
            连接是否成功
        """
        if config is None:
            config = self.server_config
            
        if not config:
            print("错误: 未加载配置。")
            return False
            
        # 查找指定名称的服务器
        server_config = None
        for server in config.get('mcp_servers', []):
            if server.get('name') == server_name:
                server_config = server
                break
                
        if not server_config:
            print(f"错误: 找不到名为 '{server_name}' 的服务器配置。")
            return False
            
        # 检查是否已连接
        if server_name in self.connections and self.connections[server_name].connected:
            print(f"已经连接到服务器 '{server_name}'。")
            # 更新当前连接指针
            self.current_connection = self.connections[server_name]
            return True
            
        # 创建新连接
        connection = ServerConnection(server_config)
        
        try:
            print(f"正在连接到服务器 '{server_name}'...")
            if await connection.connect():
                # 保存连接并更新当前连接指针
                self.connections[server_name] = connection
                self.current_connection = connection
                print(f"成功连接到服务器 '{server_name}'。")
                return True
        except Exception as e:
            print(f"连接到服务器 '{server_name}' 时出错: {str(e)}")
            await connection.cleanup()
            
        return False

    async def connect_all_default_servers(self, config: Optional[Dict] = None) -> bool:
        """连接到配置文件中标记为默认的所有服务器
        
        Args:
            config: 可选的配置字典，如果未提供则使用已加载的配置
            
        Returns:
            是否至少有一个连接成功
        """
        if config is None:
            config = self.server_config
            
        if not config:
            print("错误: 未加载配置。")
            return False
            
        # 获取默认服务器列表
        default_servers = config.get('default_server', [])
        
        # 确保默认服务器是列表
        if isinstance(default_servers, str):
            default_servers = [default_servers]
            
        if not default_servers:
            print("警告: 配置文件中未指定默认服务器。")
            return False
            
        # 尝试连接所有默认服务器
        success = False
        
        for server_name in default_servers:
            if await self.connect_to_server_by_name(server_name, config):
                success = True
                
        if not success:
            print("错误: 无法连接到任何默认服务器。")
            
        return success

    async def disconnect_from_server(self, server_name: str) -> bool:
        """断开与指定服务器的连接
        
        Args:
            server_name: 服务器名称
            
        Returns:
            断开连接是否成功
        """
        if server_name not in self.connections:
            print(f"错误: 未连接到服务器 '{server_name}'。")
            return False
            
        connection = self.connections[server_name]
        
        try:
            print(f"正在断开与服务器 '{server_name}' 的连接...")
            await connection.cleanup()
            
            # 如果当前连接是被断开的连接，重置当前连接
            if self.current_connection and self.current_connection == connection:
                self.current_connection = None
                
            # 从字典中移除连接
            del self.connections[server_name]
            
            print(f"已断开与服务器 '{server_name}' 的连接。")
            return True
        except Exception as e:
            print(f"断开与服务器 '{server_name}' 的连接时出错: {str(e)}")
            return False

    async def cleanup(self):
        """清理所有连接和资源"""
        print(f"开始清理 {len(self.connections)} 个连接...")
        
        # 复制键列表，避免在迭代过程中修改字典
        connection_names = list(self.connections.keys())
        
        # 使用更宽松的错误处理策略
        for name in connection_names:
            try:
                # 设置较短的超时时间，避免清理过程阻塞太久
                await asyncio.wait_for(self.disconnect_from_server(name), timeout=3.0)
            except asyncio.TimeoutError:
                print(f"警告: 清理连接 '{name}' 超时")
            except Exception as e:
                print(f"清理连接 '{name}' 时出错: {str(e)}")
        
        # 重置状态
        self.connections = {}
        self.current_connection = None
        
        # 给事件循环一点时间完成后台清理
        await asyncio.sleep(0.1)
        
        return True

    def list_connections(self):
        """列出所有当前连接的状态"""
        if not self.connections:
            print("当前没有活动的连接。使用 'connect <服务器名>' 连接到服务器。")
            return
            
        print("\n当前连接:")
        for name, connection in self.connections.items():
            status = "已连接" if connection.connected else "已断开"
            is_current = " (当前)" if connection == self.current_connection else ""
            
            print(f"- {name}: {status}{is_current}")
            
            # 显示连接的工具/资源/提示数量
            num_tools = len(connection.tools_cache)
            num_resources = len(connection.resources_cache)
            num_prompts = len(connection.prompts_cache)
            
            details = []
            if num_tools > 0:
                details.append(f"{num_tools} 个工具")
            if num_resources > 0:
                details.append(f"{num_resources} 个资源")
            if num_prompts > 0:
                details.append(f"{num_prompts} 个提示模板")
                
            if details:
                print(f"  可用: {', '.join(details)}")

    def switch_current_connection(self, server_name: str) -> bool:
        """切换当前活动的连接
        
        Args:
            server_name: 服务器名称
            
        Returns:
            切换是否成功
        """
        if server_name not in self.connections:
            print(f"错误: 未连接到服务器 '{server_name}'。")
            return False
            
        connection = self.connections[server_name]
        
        if not connection.connected:
            print(f"警告: 服务器 '{server_name}' 已断开连接。")
            return False
            
        self.current_connection = connection
        print(f"已切换到服务器 '{server_name}'。")
        return True

    def collect_all_tools(self) -> Dict[str, List[Dict]]:
        """收集所有连接的工具信息
        
        Returns:
            工具信息字典，键为服务器名称
        """
        result = {}
        
        for name, connection in self.connections.items():
            if connection.connected and connection.tools_cache:
                result[name] = connection.tools_cache
                
        return result

    def collect_all_resources(self) -> Dict[str, List[Dict]]:
        """收集所有连接的资源信息
        
        Returns:
            资源信息字典，键为服务器名称
        """
        result = {}
        
        for name, connection in self.connections.items():
            if connection.connected and connection.resources_cache:
                result[name] = connection.resources_cache
                
        return result

    def collect_all_prompts(self) -> Dict[str, List[Dict]]:
        """收集所有连接的提示模板信息
        
        Returns:
            提示模板信息字典，键为服务器名称
        """
        result = {}
        
        for name, connection in self.connections.items():
            if connection.connected and connection.prompts_cache:
                result[name] = connection.prompts_cache
                
        return result

    def list_all_tools(self):
        """列出所有连接的可用工具"""
        all_tools = self.collect_all_tools()
        
        if not all_tools:
            print("没有连接的服务器提供工具。")
            return
            
        print("\n可用工具:")
        for server_name, tools in all_tools.items():
            print(f"\n{server_name}:")
            
            for tool in tools:
                name = tool.get("name", "未知工具")
                description = tool.get("description", "无描述")
                print(f"- {name}: {description}")
                
                # 显示参数信息
                schema = tool.get("schema", {})
                if isinstance(schema, dict) and "properties" in schema:
                    properties = schema.get("properties", {})
                    required = schema.get("required", [])
                    
                    if properties:
                        print("  参数:")
                        for param_name, param_info in properties.items():
                            param_type = param_info.get("type", "any")
                            param_desc = param_info.get("description", "")
                            req_marker = "*" if param_name in required else ""
                            
                            print(f"  - {param_name}{req_marker} ({param_type}): {param_desc}")

    def list_all_resources(self):
        """列出所有连接的可用资源"""
        all_resources = self.collect_all_resources()
        
        if not all_resources:
            print("没有连接的服务器提供资源。")
            return
            
        print("\n可用资源:")
        for server_name, resources in all_resources.items():
            print(f"\n{server_name}:")
            
            for resource in resources:
                uri = resource.get("uri", "未知资源")
                description = resource.get("description", "无描述")
                print(f"- {uri}: {description}")

    def list_all_prompts(self):
        """列出所有连接的可用提示模板"""
        all_prompts = self.collect_all_prompts()
        
        if not all_prompts:
            print("没有连接的服务器提供提示模板。")
            return
            
        print("\n可用提示模板:")
        for server_name, prompts in all_prompts.items():
            print(f"\n{server_name}:")
            
            for prompt in prompts:
                name = prompt.get("name", "未知提示模板")
                description = prompt.get("description", "无描述")
                print(f"- {name}: {description}")
                
                # 显示参数信息
                schema = prompt.get("schema", {})
                if isinstance(schema, dict) and "properties" in schema:
                    properties = schema.get("properties", {})
                    required = schema.get("required", [])
                    
                    if properties:
                        print("  参数:")
                        for param_name, param_info in properties.items():
                            param_type = param_info.get("type", "any")
                            param_desc = param_info.get("description", "")
                            req_marker = "*" if param_name in required else ""
                            
                            print(f"  - {param_name}{req_marker} ({param_type}): {param_desc}")

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
            raise ValueError(f"未连接到服务器: {server_name}")
            
        connection = self.connections[server_name]
        # 直接调用工具并返回原始结果，不进行JSON序列化
        result = await connection.call_tool(tool_name, params or {})
        return result  # 直接返回结果，不尝试序列化

    async def get_resource_by_server(self, server_name: str, resource_uri: str) -> Any:
        """获取指定服务器上的资源
        
        Args:
            server_name: 服务器名称
            resource_uri: 资源URI
            
        Returns:
            资源内容
        """
        if server_name not in self.connections:
            raise ValueError(f"错误: 未连接到服务器 '{server_name}'。")
            
        connection = self.connections[server_name]
        
        if not connection.connected:
            raise ValueError(f"错误: 服务器 '{server_name}' 已断开连接。")
            
        return await connection.get_resource(resource_uri)

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
            raise ValueError(f"错误: 未连接到服务器 '{server_name}'。")
            
        connection = self.connections[server_name]
        
        if not connection.connected:
            raise ValueError(f"错误: 服务器 '{server_name}' 已断开连接。")
            
        return await connection.call_prompt(prompt_name, params)

    async def smart_query(self, query: str) -> str:
        """处理LLM回答
        
        Args:
            query: 用户回答
            
        Returns:
            处理后的结果
        """
        if not self.llm:
            return "错误: LLM未初始化，无法执行回答处理。请检查环境变量配置。"
            
        return await self.llm_processor.smart_query(query)

    async def interactive_mode(self):
        """交互式命令行模式"""
        from .cli import run_interactive_cli
        await run_interactive_cli(self)

    def list_server_tools(self, server_name: str):
        """列出指定服务器的可用工具
        
        Args:
            server_name: 服务器名称
        """
        if server_name not in self.connections:
            print(f"错误: 未连接到服务器 '{server_name}'")
            return
        
        connection = self.connections[server_name]
        if not connection.connected:
            print(f"警告: 服务器 '{server_name}' 连接已断开")
            return
        
        tools = connection.tools_cache
        if not tools:
            print(f"服务器 '{server_name}' 没有可用的工具")
            return
        
        print(f"\n{server_name} 可用工具:")
        for tool in tools:
            name = tool.get("name", "未知工具")
            description = tool.get("description", "无描述")
            print(f"- {name}: {description}")
            
            # 显示参数信息
            schema = tool.get("schema", {})
            if isinstance(schema, dict) and "properties" in schema:
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                
                if properties:
                    print("  参数:")
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        req_marker = "*" if param_name in required else ""
                        
                        print(f"  - {param_name}{req_marker} ({param_type}): {param_desc}")

    def list_server_resources(self, server_name: str):
        """列出指定服务器的可用资源
        
        Args:
            server_name: 服务器名称
        """
        if server_name not in self.connections:
            print(f"错误: 未连接到服务器 '{server_name}'")
            return
        
        connection = self.connections[server_name]
        if not connection.connected:
            print(f"警告: 服务器 '{server_name}' 连接已断开")
            return
        
        resources = connection.resources_cache
        if not resources:
            print(f"服务器 '{server_name}' 没有可用的资源")
            return
        
        print(f"\n{server_name} 可用资源:")
        for resource in resources:
            uri = resource.get("uri", "未知资源")
            content_type = resource.get("contentType", "未知类型")
            description = resource.get("description", "无描述")
            print(f"- {uri} ({content_type}): {description}")

    def list_server_prompts(self, server_name: str):
        """列出指定服务器的可用提示模板
        
        Args:
            server_name: 服务器名称
        """
        if server_name not in self.connections:
            print(f"错误: 未连接到服务器 '{server_name}'")
            return
        
        connection = self.connections[server_name]
        if not connection.connected:
            print(f"警告: 服务器 '{server_name}' 连接已断开")
            return
        
        prompts = connection.prompts_cache
        if not prompts:
            print(f"服务器 '{server_name}' 没有可用的提示模板")
            return
        
        print(f"\n{server_name} 可用提示模板:")
        for prompt in prompts:
            name = prompt.get("name", "未知提示模板")
            description = prompt.get("description", "无描述")
            print(f"- {name}: {description}")
            
            # 显示参数信息
            schema = prompt.get("schema", {})
            if isinstance(schema, dict) and "properties" in schema:
                properties = schema.get("properties", {})
                required = schema.get("required", [])
                
                if properties:
                    print("  参数:")
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        req_marker = "*" if param_name in required else ""
                        
                        print(f"  - {param_name}{req_marker} ({param_type}): {param_desc}") 