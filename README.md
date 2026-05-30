# GPT Image2 多功能抽卡器

一个面向 GPT Image 2 / OpenAI Images 兼容接口的本地图像生成工作台。它把提示词创作、批量出图、随机创意、多轮视觉迭代和图片提示词反推整合在一个 Gradio 界面里，适合需要更细粒度控制图片数量、比例、分辨率和并发流程的用户。

## 为什么做这个工具

很多 API 中转站已经提供了 GPT Image 2 或类似图片生成模型的接口，但常见 AI 聊天软件对图片生成参数的支持并不完整，例如不方便设置生成数量、比例、分辨率，也缺少批量生成、随机生成和自动迭代能力。

这个应用就是为这些场景设计的：通过一个本地可运行的界面，把图片接口、文本模型和多模态模型串起来，让普通用户也可以更灵活地做图像创作实验。

本项目由非程序员用户在 Codex 和 GPT-5.5 的帮助下完成，代码主要由 AI 自动生成。欢迎感兴趣的用户克隆、修改、继续改进。

## 功能亮点

- **手动模式**：输入固定提示词，选择数量、比例和分辨率后批量生成。
- **随机模式**：调用文本模型生成一段随机提示词，再用这段提示词生成多张图片。
- **创意模式**：同时生成多段不同随机提示词，并以可控并发流水线出图。
- **自我迭代模式**：生成图片后调用多模态模型进行视觉评估，自动返回优化提示词进入下一轮。
- **提示词反推**：上传图片或填写本地图片路径，由多模态模型反推出中文图像提示词。
- **统一设置页**：集中管理图片生成接口、文本模型接口、多模态模型接口、重试策略、保存目录和提示词模板。
- **协议适配**：支持 OpenAI Chat、OpenAI Responses、Gemini 原生协议、Claude Messages。
- **思考档位**：文本模型和多模态模型可以分别设置思考强度，并自动映射到不同厂商的参数。
- **图片压缩**：发送给多模态模型前自动压缩图片，降低上传体积，提高评估速度。
- **错误提示与重试**：接口报错、连接断开、重试状态会显示在状态栏中。

## 界面预览

可以在项目目录放入以下截图文件，GitHub 页面会显示为模式预览：

```text
screenshot_manual.png
screenshot_random.png
screenshot_creative.png
screenshot_iterative.png
```

建议截图内容分别对应手动模式、随机模式、创意模式和自我迭代模式。

## 安装

建议使用 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
```

## 启动

```bash
python app.py
```

启动后会自动打开系统默认浏览器。Windows 用户也可以双击：

```text
启动.bat
```

## 基本使用

1. 打开“设置”页。
2. 填写图片生成接口的 API 地址、模型 ID 和 API Key。
3. 如需随机模式、创意模式，填写文本模型接口。
4. 如需自我迭代或提示词反推，填写多模态模型接口。
5. 点击“保存设置”。
6. 回到对应模式开始生成。

刷新页面后，如果设置显示不一致，可以点击“重新读取设置”。重新运行脚本也会读取已保存的配置。

## 接口说明

### 图片生成接口

用于实际生成图片。通常需要 OpenAI Images 兼容接口。

建议输入格式：

```text
https://example.com
```

或完整接口：

```text
https://example.com/v1/images/generations
```

### 文本模型接口

用于随机提示词生成和创意模式提示词生成。

支持：

- OpenAI Chat Completions
- OpenAI Responses
- Gemini 原生协议
- Claude Messages

### 多模态模型接口

用于自我迭代视觉评估和提示词反推。

支持：

- OpenAI Chat Completions
- OpenAI Responses
- Gemini 原生协议
- Claude Messages

## 思考档位

文本模型和多模态模型都可以单独设置：

```text
关闭 / 低 / 中 / 高 / 最高
```

程序会根据协议自动映射：

- OpenAI Chat：`reasoning_effort`
- OpenAI Responses：`reasoning.effort`
- Gemini：`thinkingConfig`
- Claude：`thinking` 或 adaptive thinking

如果某个接口不支持思考参数，选择“关闭”即可。

## 图片压缩

自我迭代和提示词反推会在发送图片给多模态模型前创建一份内存中的压缩 JPEG：

- 不影响本地保存的原图
- 默认长边限制为 1536px
- 默认 JPEG 质量为 90

这样可以明显减少多模态请求体积。

## 配置文件

首次保存设置后，程序会生成：

```text
app_config.json
```

它用于保存接口地址、模型 ID、保存目录、重试设置和提示词模板。可以参考：

```text
app_config.example.json
```

## 目录说明

```text
app.py                    主程序
config_store.py           配置读写工具
requirements.txt          Python 依赖
DEPENDENCIES.md           依赖说明
app_config.example.json   配置示例
启动.bat                  Windows 启动脚本
```

## 参与改进

欢迎提交 issue、建议或改进版本。可以继续扩展的方向包括：

- 更多图片生成协议
- 更细的任务队列管理
- 更灵活的提示词模板系统
- 更强的批量结果筛选和收藏
- 多模型对比生成
