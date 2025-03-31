# Lite-MCP-Client

## 📝 项目介绍

Lite-MCP-Client是一个基于命令行的轻量级MCP客户端工具，可以连接到多种MCP（Model-Chat-Prompt）服务器，帮助用户轻松调用服务器提供的工具、资源和提示模板。该客户端支持与大型语言模型（如OpenAI的GPT和Google的Gemini）集成，实现智能化查询和处理。

![image](https://img.pub/p/a217fd508f77b65ac7bb.png)

![image](https://img.pub/p/7d38d7109f5fe7bada1f.png)

## ✨ 主要特点

- **多服务器连接管理**：同时连接并管理多个MCP服务器
- **支持多种服务器类型**：兼容STDIO和SSE类型的服务器
- **工具调用**：轻松调用服务器提供的各种工具
- **资源获取**：访问服务器提供的资源
- **提示模板使用**：使用服务器定义的提示模板
- **智能查询**：通过自然语言查询，自动决定使用哪些工具和资源
- **灵活的交互模式**：支持命令行参数和交互式终端界面
- **智能化对话历史管理**：维护会话上下文，实现连续对话

## 🚀 安装指南

### 前提条件

- Python 3.10+
- uv

### 安装步骤

1. 克隆仓库到本地：

```bash
git clone https://github.com/sligter/lite-mcp-client.git
```

2. 安装依赖：

```bash
uv sync
```

3. 配置环境变量：


```
cp .env.example .env
```

## 📖 使用说明

### 基本使用

```bash
# 启动交互式模式
uv run client.py --interactive

# 使用特定服务器
uv run client.py --server "服务器名称"

# 连接所有默认服务器
uv run client.py --connect-all

# 执行智能查询
uv run client.py --query "查询微博热点新闻并总结"
# 或者直接
uv run client.py "查询微博热点新闻并总结"

# 调用特定工具
uv run client.py --call "服务器名.工具名" {"参数1": "值1"}

# 获取资源
uv run client.py --get "服务器名.资源URI"

# 使用提示模板
uv run client.py --prompt "服务器名.提示名" {"参数1": "值1"}
```

### 配置文件

默认配置文件为`mcp_config.json`，格式如下：

```json
{
    "mcp_servers": [
      {
        "name": "各平台热搜查询",
        "type": "stdio",
        "command": "uvx",
        "args": ["mcp-newsnow"],
        "env": {},
        "description": "热点话题查询"
      },
      {
        "name": "Fetch",
        "type": "stdio",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "env": {},
        "description": "访问指定链接"
      },
      {
        "name": "其他服务",
        "type": "sse",
        "url": "http://localhost:3000/sse",
        "headers": {},
        "description": "其他服务描述"
      }
    ],
    "default_server": ["各平台热搜查询", "Fetch"]
}
```

### 交互式命令

在交互式模式下，支持以下命令：

- `connect <服务器名>` - 连接到指定服务器
- `connect-all` - 连接到所有默认服务器
- `disconnect <服务器名>` - 断开与指定服务器的连接
- `switch <服务器名>` - 切换到已连接的服务器
- `connections` - 列出所有连接
- `tools` - 列出所有已连接服务器的工具
- `resources` - 列出所有已连接服务器的资源
- `prompts` - 列出所有已连接服务器的提示模板
- `call <服务器名>.<工具名> [参数JSON]` - 调用指定服务器上的工具
- `get <服务器名>.<资源URI>` - 获取指定服务器上的资源
- `prompt <服务器名>.<提示名> [参数JSON]` - 使用指定服务器上的提示模板
- `ask <自然语言提问>` - 智能处理提问，自动调用所需工具
- `clear-history` - 清除对话历史，开始新对话
- `help` - 显示帮助信息
- `quit` - 退出程序

## 🌟 实际应用示例

### 智能新闻聚合

```bash
uv run client.py "获取今天的热门新闻并按主题分类"
```

### 多服务端

```bash
uv run client.py "先从财经服务获取股市数据，然后使用AI分析服务生成投资建议"
```

## 🔧 高级配置

### 服务器配置选项

| 参数 | 类型 | 描述 |
|------|------|------|
| name | 字符串 | 服务器名称 |
| type | 字符串 | 服务器类型（"stdio"或"sse"） |
| command | 字符串 | 启动STDIO服务器的命令 |
| args | 列表 | 命令行参数 |
| env | 对象 | 环境变量 |
| url | 字符串 | SSE服务器URL |
| headers | 对象 | HTTP请求头 |
| description | 字符串 | 服务器描述 |

## 📚 依赖项

- `asyncio`: 异步IO支持
- `mcp`: MCP协议客户端库
- `langchain_openai`: OpenAI模型集成
- `langchain_google_genai`: Google生成式AI模型集成
- `dotenv`: 环境变量管理
- `json`: JSON数据处理

## 🤝 贡献指南

欢迎提交问题报告和功能请求！如果您想贡献代码：

1. Fork此仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建一个Pull Request

## 📄 许可证

此项目采用MIT许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢所有MCP协议的贡献者
- 感谢语言模型提供商的API支持
- 感谢开源社区的支持和贡献

---

*注：本客户端仅提供与MCP服务的接口，具体功能取决于所连接服务器提供的工具和资源。*
