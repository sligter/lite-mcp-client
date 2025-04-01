"""配置文件加载和管理模块"""

import json
import sys
from pathlib import Path
from typing import Dict

# 默认配置文件路径
DEFAULT_CONFIG_PATH = "mcp_config.json"

def load_server_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict:
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
        _create_default_config(config_file)

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

def _create_default_config(config_file: Path):
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
            }
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

def list_available_servers(config: Dict) -> None:
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