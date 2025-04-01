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

def parse_server_and_name(spec: str, default_server: Optional[str] = None) -> Tuple[str, str]:
    """解析格式为'server.name'的字符串
    
    Args:
        spec: 格式为'server.name'的字符串
        default_server: 如果未指定服务器时的默认值
        
    Returns:
        (服务器名, 名称)的元组
    """
    if '.' in spec:
        server_name, name = spec.split('.', 1)
        return server_name, name
    elif default_server:
        return default_server, spec
    else:
        raise ValueError(f"无法解析 '{spec}'，格式应为 'server.name'，且无默认服务器")

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