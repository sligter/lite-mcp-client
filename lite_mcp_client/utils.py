"""工具函数模块"""

import json
import re
from typing import Dict, Any, Tuple, Optional

def parse_json_params(params_str: str) -> Dict:
    """解析JSON参数字符串
    
    Args:
        params_str: 包含JSON的字符串
        
    Returns:
        解析后的参数字典
    """
    if not params_str:
        return {}
        
    try:
        return json.loads(params_str)
    except json.JSONDecodeError:
        # 尝试清理并重新解析
        cleaned = clean_json_string(params_str)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析JSON参数: {e}")

def clean_json_string(text: str) -> str:
    """清理JSON字符串，修复常见格式问题
    
    Args:
        text: 待清理的JSON字符串
        
    Returns:
        清理后的JSON字符串
    """
    # 移除可能的Markdown格式
    text = re.sub(r'```.*\n|```', '', text)
    
    # 移除多余的空格和换行
    text = text.strip()
    
    # 替换不规范的JSON语法
    text = re.sub(r'(\w+):', r'"\1":', text)  # 将 key: 替换为 "key":
    text = re.sub(r',\s*}', '}', text)        # 移除对象末尾的逗号
    text = re.sub(r',\s*]', ']', text)        # 移除数组末尾的逗号
    
    return text

def parse_server_and_name(spec: str, client=None) -> Tuple[str, str]:
    """解析服务器名和工具/提示/资源名
    
    Args:
        spec: 格式为 "服务器名.名称" 或仅 "名称" 的字符串
        client: 客户端实例，用于获取当前连接
        
    Returns:
        包含服务器名和名称的元组
        
    Raises:
        ValueError: 如果格式无效或缺少当前连接
    """
    if '.' in spec:
        # 如果提供了完整的 "服务器名.名称" 格式
        server_name, name = spec.split('.', 1)
        return server_name, name
    else:
        # 如果只提供了名称，使用当前连接的服务器
        if client is None or client.current_connection is None:
            raise ValueError("未指定服务器名，且没有活动的服务器连接。请使用 '服务器名.名称' 格式或先切换到一个活动服务器。")
        return client.current_connection.name, spec

def format_tool_help(tool_info: Dict) -> str:
    """格式化工具帮助信息
    
    Args:
        tool_info: 工具信息字典
        
    Returns:
        格式化的帮助信息字符串
    """
    name = tool_info.get("name", "未知工具")
    description = tool_info.get("description", "无描述")
    
    lines = [f"工具: {name}", f"描述: {description}"]
    
    schema = tool_info.get("schema", {})
    if isinstance(schema, dict) and "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        if properties:
            lines.append("\n参数:")
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                req_marker = "* " if param_name in required else ""
                
                lines.append(f"  - {req_marker}{param_name} ({param_type}): {param_desc}")
                
    return "\n".join(lines)

def print_help_message():
    """打印帮助信息"""
    print("\n可用命令:")
    print("  connect <服务器名>       - 连接到配置文件中定义的指定服务器")
    print("  connect-all             - 连接到配置文件中所有标记为默认的服务器")
    print("  disconnect <服务器名>    - 断开与指定服务器的连接并清理资源")
    print("  switch <服务器名>        - 切换当前活动的服务器连接")
    print("  connections (conn)      - 列出所有当前连接及其状态")
    print("  tools [服务器名]         - 列出所有或指定服务器的可用工具")
    print("  resources (res) [服务器名] - 列出所有或指定服务器的可用资源")
    print("  prompts [服务器名]       - 列出所有或指定服务器的可用提示模板")
    print("  call <srv.tool> [参数]   - 调用指定服务器上的工具")
    print("  call <tool> [参数]       - 调用当前服务器上的工具")
    print("  get <srv.uri>           - 获取指定服务器上的资源")
    print("  get <uri>               - 获取当前服务器上的资源")
    print("  prompt <srv.prompt> [参数] - 使用指定服务器上的提示模板")
    print("  prompt <prompt> [参数]   - 使用当前服务器上的提示模板")
    print("  ask <自然语言问题>        - LLM处理提问，自动选择并调用工具")
    print("  clear-history (clh)      - 清除 'ask' 命令的对话历史记录")
    print("  help                    - 显示此帮助信息")
    print("  quit / exit             - 退出程序") 