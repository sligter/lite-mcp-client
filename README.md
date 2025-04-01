# LiteMCP 客户端

一个轻量级的 MCP (Machine Communication Protocol) 客户端，可以连接到任意 MCP 服务器并使用其功能。

## 功能特点

- 支持同时连接多个 MCP 服务器
- 支持 STDIO 和 SSE 类型的服务器连接
- 提供简单的命令行界面和交互模式
- 支持 LLM 智能回答处理和工具调用
- 支持工具调用、资源获取和提示模板使用

## 安装

```bash
# 基本安装
pip install lite-mcp-client

# 如果需要LLM功能，使用[llm]扩展
pip install "lite-mcp-client[llm]"
```

## 快速开始

1. 创建配置文件 `mcp_config.json`（如不存在，将创建默认配置）
2. 运行客户端：

```bash
litemcp

# 或者指定配置文件
litemcp -c path/to/config.json

# 直接连接特定服务器
litemcp --server 服务器名
```

## 命令行用法

```
litemcp --help
```

### 常用命令

- `litemcp --call 服务器名.工具名 --params '{"参数": "值"}'` - 调用工具
- `litemcp --get 服务器名.资源URI` - 获取资源
- `litemcp --prompt 服务器名.提示名 --params '{"参数": "值"}'` - 使用提示模板
- `litemcp --query "自然语言回答"` - 使用LLM处理回答
- `litemcp -i` - 进入交互模式

## 交互模式

在交互模式下，可以使用以下命令：

- `connect <服务器名>` - 连接到服务器
- `connect-all` - 连接到所有默认服务器
- `disconnect <服务器名>` - 断开连接
- `tools` - 列出可用工具
- `call <服务器名.工具名> {JSON参数}` - 调用工具
- `ask <自然语言回答>` - 智能回答处理

## LLM 配置

复制 `.env.example` 到 `.env` 并填入你的API密钥：

```bash
cp .env.example .env
```

## 许可证

MIT
