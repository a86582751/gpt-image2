import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
import time
import requests
import gradio as gr
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse
from config_store import load_config, update_config

BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_ID = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
RANDOM_BASE_URL = os.getenv("RANDOM_PROMPT_BASE_URL", "").rstrip("/")
RANDOM_API_KEY = os.getenv("RANDOM_PROMPT_API_KEY", "")
RANDOM_MODEL_ID = os.getenv("RANDOM_PROMPT_MODEL", "")
ITERATION_BASE_URL = os.getenv("ITERATION_BASE_URL", "").rstrip("/")
ITERATION_API_KEY = os.getenv("ITERATION_API_KEY", "")
ITERATION_MODEL_ID = os.getenv("ITERATION_MODEL", "")
QUALITY_PRESETS = ["auto", "medium", "high", "low"]
MODEL_PROTOCOL_PRESETS = ["自动识别", "OpenAI Chat", "OpenAI Responses", "Gemini 原生", "Claude Messages"]
ITERATION_PROMPT_SOURCE_PRESETS = ["随机提示词", "自定义提示词"]
REASONING_EFFORT_PRESETS = ["关闭", "低", "中", "高", "最高"]
STOP_FLAGS = {
    "manual": False,
    "random": False,
    "creative": False,
    "iterative": False,
    "reverse": False,
}

RANDOM_SYSTEM_PROMPT = """你是一位专业的AI视觉创意提示词工程师。你的任务是生成一段适合AI图像生成的高质量中文关键词组合，强调电影感、艺术构图、空间层次、细腻光影和富有故事感的视觉氛围。

**格式规则：**
1. 必须使用中文逗号`，`分隔关键词和短语。
2. 必须以核心质量词开头，如：`杰作，最佳质量，超高细节，电影感摄影`。
3. 内容顺序应大致遵循：`核心质量 -> 主体 -> 外貌或物体特征 -> 服装或材质 -> 姿态或动作 -> 场景 -> 光影 -> 色彩 -> 画风`。
4. 严禁输出解释、段落说明或无关文字；只输出可直接用于图像生成的提示词。"""

RANDOM_USER_PROMPT = """请创建一段随机图像生成提示词。

要求：
（突出电影感、艺术性、空间层次、情绪氛围和故事感）
（可以包含人物、风景、建筑、静物或幻想场景）
（场景应具体、鲜明，避免空泛描述）
（不要出现敏感、露骨、血腥、仇恨或违法内容）
（输出必须是中文关键词组合，不要解释）"""

ITERATION_OPTIMIZER_PROMPT = """你是一位专业的AI图像视觉评估师与提示词优化专家。我将为你提供一张AI生成图片，以及生成该图片所使用的提示词。

你的任务是：审视这张图片，判断它在主体表达、构图、空间层次、光影氛围、色彩关系、材质细节和故事感上的可提升空间，并输出一段全新优化过的中文提示词，让下一轮生成的图片更具电影感、艺术表现力和视觉完成度。

**评估与优化方向（在脑内进行，不要写出来）：**
1. 主体与构图：强化主体辨识度、姿态或物体形态、视线引导、前中后景层次。
2. 场景与叙事：让环境更具体，增加能够激发想象的故事线索和空间细节。
3. 光影与色彩：加入更明确的光源、时间、色温、阴影层次、胶片或摄影质感。
4. 画面质感：增强材质、纹理、镜头语言和艺术风格的一致性。

**输出要求：**
你只能输出一串用于AI绘画的中文词组。
必须使用中文逗号分隔。
顺序遵循：核心质量 -> 主体特征 -> 细节与材质 -> 动作或构图 -> 环境背景 -> 光影色彩 -> 画风质感。
不要输出评估过程、解释、标题或无关文字。"""

REVERSE_PROMPT = """你是一位专业的AI图像生成提示词工程师。请仔细观察这张图像，并反推出一段可用于AI图像生成的中文提示词。

请覆盖主体、前景、中景、背景、构图、视觉引导、材质、色彩、光影氛围、镜头语言、艺术风格和画面质感等细节，让提示词具有深度、氛围和艺术感。

要求：只输出中文提示词，不要描述水印、签名、边框或无关文字，不要解释，不要总结，不要添加标题或符号，限制在800字以内。"""

ASPECT_RATIOS = {
    "1:1 正方形": {
        "标准": "1248x1248",
        "高清": "2048x2048",
        "超清": "2880x2880",
    },
    "4:3 横图": {
        "标准": "1440x1072",
        "高清": "2048x1536",
        "超清": "3264x2448",
    },
    "3:2 横图": {
        "标准": "1536x1024",
        "高清": "2160x1440",
        "超清": "3546x2304",
    },
    "16:9 宽屏": {
        "标准": "1664x928",
        "高清": "2560x1440",
        "超清": "3840x2160",
    },
}

RESOLUTION_PRESETS = {
    "标准": "标准",
    "高清": "高清",
    "超清": "超清",
}

DEFAULT_CONFIG = {
    "prompt": "",
    "save_dir": "",
    "image_count": 1,
    "concurrency": 1,
    "text_concurrency": 10,
    "image_concurrency": 3,
    "creative_count": 5,
    "retry_count": 1,
    "retry_delay": 2,
    "aspect_ratio": "4:3 横图",
    "resolution": "高清",
    "base_url": BASE_URL,
    "model_id": MODEL_ID,
    "quality": "auto",
    "api_key": API_KEY,
    "random_base_url": RANDOM_BASE_URL,
    "random_model_id": RANDOM_MODEL_ID,
    "random_api_key": RANDOM_API_KEY,
    "random_protocol": "自动识别",
    "random_reasoning_effort": "最高",
    "random_preference": "",
    "iteration_count": 3,
    "iteration_prompt_source": "随机提示词",
    "iteration_custom_prompt": "",
    "iteration_base_url": ITERATION_BASE_URL,
    "iteration_model_id": ITERATION_MODEL_ID,
    "iteration_api_key": ITERATION_API_KEY,
    "iteration_protocol": "自动识别",
    "iteration_reasoning_effort": "关闭",
    "random_system_prompt": RANDOM_SYSTEM_PROMPT,
    "random_user_prompt": RANDOM_USER_PROMPT,
    "iteration_optimizer_prompt": ITERATION_OPTIMIZER_PROMPT,
    "reverse_prompt": REVERSE_PROMPT,
}

def normalize_config(config):
    config = config.copy()
    if config["aspect_ratio"] not in ASPECT_RATIOS:
        config["aspect_ratio"] = DEFAULT_CONFIG["aspect_ratio"]
    if config["resolution"] not in RESOLUTION_PRESETS:
        config["resolution"] = DEFAULT_CONFIG["resolution"]
    if config["quality"] not in QUALITY_PRESETS:
        config["quality"] = DEFAULT_CONFIG["quality"]
    if config["random_protocol"] not in MODEL_PROTOCOL_PRESETS:
        config["random_protocol"] = DEFAULT_CONFIG["random_protocol"]
    if config["iteration_protocol"] not in MODEL_PROTOCOL_PRESETS:
        config["iteration_protocol"] = DEFAULT_CONFIG["iteration_protocol"]
    if config["iteration_prompt_source"] not in ITERATION_PROMPT_SOURCE_PRESETS:
        config["iteration_prompt_source"] = DEFAULT_CONFIG["iteration_prompt_source"]
    if config["random_reasoning_effort"] not in REASONING_EFFORT_PRESETS:
        config["random_reasoning_effort"] = DEFAULT_CONFIG["random_reasoning_effort"]
    if config["iteration_reasoning_effort"] not in REASONING_EFFORT_PRESETS:
        config["iteration_reasoning_effort"] = DEFAULT_CONFIG["iteration_reasoning_effort"]
    return config


CONFIG = normalize_config(load_config(DEFAULT_CONFIG))

CONNECT_TIMEOUT = 30
TEXT_READ_TIMEOUT = 300
VISION_READ_TIMEOUT = 600
IMAGE_READ_TIMEOUT = 1200
VISION_IMAGE_MAX_SIDE = 1536
VISION_IMAGE_JPEG_QUALITY = 90


def persist_config(updates):
    """Persist UI state and keep the in-memory config aligned for later callbacks."""
    update_config(DEFAULT_CONFIG, updates)
    CONFIG.update(updates)


def resolve_size(aspect_ratio, resolution):
    ratio_sizes = ASPECT_RATIOS.get(aspect_ratio, ASPECT_RATIOS["4:3 横图"])
    resolution_key = RESOLUTION_PRESETS.get(resolution, "高清")
    return ratio_sizes[resolution_key]


def normalize_quality(quality):
    quality = (quality or "auto").strip()
    return quality if quality in QUALITY_PRESETS else "auto"


def normalize_protocol(protocol):
    protocol = (protocol or "自动识别").strip()
    return protocol if protocol in MODEL_PROTOCOL_PRESETS else "自动识别"


def normalize_iteration_prompt_source(source):
    source = (source or "随机提示词").strip()
    return source if source in ITERATION_PROMPT_SOURCE_PRESETS else "随机提示词"


def normalize_reasoning_effort(effort):
    effort = (effort or "关闭").strip()
    return effort if effort in REASONING_EFFORT_PRESETS else "关闭"


def update_iteration_source_ui(source):
    source = normalize_iteration_prompt_source(source)
    is_custom = source == "自定义提示词"
    preference_label = "创作主题" if is_custom else "初始创作方向"
    preference_placeholder = "例如：雨后街角、玻璃花房、黄昏海岸" if is_custom else "例如：雨后街角、玻璃花房、黄昏海岸"
    custom_prompt_label = "初始提示词（点击输入你需要的提示词）" if is_custom else "初始提示词（由文本模型随机生成）"
    return (
        gr.update(label=preference_label, placeholder=preference_placeholder),
        gr.update(label=custom_prompt_label, interactive=is_custom),
    )


def normalize_retry_settings(retry_count, retry_delay):
    return max(0, int(retry_count)), max(0, float(retry_delay))


def request_timeout(read_timeout):
    return (CONNECT_TIMEOUT, read_timeout)


def format_protocol_label(protocol):
    return {
        "openai_chat": "OpenAI Chat",
        "openai_responses": "OpenAI Responses",
        "gemini": "Gemini 原生",
        "anthropic_messages": "Claude Messages",
    }.get(protocol, protocol)


def display_endpoint(url):
    return (url or "").split("?", 1)[0]


def format_bytes(byte_count):
    byte_count = int(byte_count or 0)
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f} KB"
    return f"{byte_count / (1024 * 1024):.1f} MB"


def gemini_headers(api_key, request_url):
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key.strip()}
    if "generativelanguage.googleapis.com" not in request_url:
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def apply_reasoning_settings(payload, protocol, effort, model_id=""):
    """Apply vendor-specific thinking/reasoning parameters only when the user enables them."""
    effort = normalize_reasoning_effort(effort)
    if effort == "关闭":
        return payload

    effort_map = {
        "低": "low",
        "中": "medium",
        "高": "high",
        "最高": "high",
    }
    max_compatible_effort = {
        "低": "low",
        "中": "medium",
        "高": "high",
        "最高": "max",
    }
    gemini_budget = {
        "低": 1024,
        "中": 4096,
        "高": 8192,
        "最高": 24576,
    }
    claude_budget = {
        "低": 1024,
        "中": 4096,
        "高": 8192,
        "最高": 16000,
    }
    claude_effort = {
        "低": "low",
        "中": "medium",
        "高": "high",
        "最高": "max",
    }

    model_lower = (model_id or "").lower()
    if protocol == "openai_responses":
        payload["reasoning"] = {"effort": effort_map[effort]}
        return payload
    if protocol == "openai_chat":
        if "deepseek" in model_lower:
            payload["reasoning_effort"] = max_compatible_effort[effort]
            payload["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            payload["reasoning_effort"] = effort_map[effort]
        return payload
    if protocol == "gemini":
        thinking_config = {"thinkingBudget": gemini_budget[effort]}
        if model_lower.startswith("gemini-3"):
            thinking_config = {"thinkingLevel": effort_map[effort]}
        payload["generationConfig"] = {**payload.get("generationConfig", {}), "thinkingConfig": thinking_config}
        return payload
    if protocol == "anthropic_messages":
        if any(model_name in model_lower for model_name in ("claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6")):
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": claude_effort[effort]}
            payload["max_tokens"] = max(int(payload.get("max_tokens", 2000)), 4096)
            return payload
        payload["thinking"] = {"type": "enabled", "budget_tokens": claude_budget[effort]}
        payload["max_tokens"] = max(int(payload.get("max_tokens", 2000)), claude_budget[effort] + 1024)
        return payload
    return payload


def resolve_api_url(base_url):
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("请填写 API 地址。")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url}"

    parsed_url = urlparse(base_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("API 地址格式不正确，请填写类似 https://example.com 的地址。")

    if base_url.endswith("/images/generations"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/images/generations"
    return f"{base_url}/v1/images/generations"


def normalize_base_url(base_url, empty_message, invalid_message):
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError(empty_message)
    if not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url}"

    parsed_url = urlparse(base_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(invalid_message)
    return base_url


def detect_text_protocol(base_url, model_id):
    base_url = normalize_base_url(base_url, "请填写文本模型 API 地址。", "文本模型 API 地址格式不正确。")
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("请填写文本模型 ID。")

    model_lower = model_id.lower()
    if base_url.endswith("/chat/completions"):
        return "openai_chat", base_url
    if base_url.endswith("/responses"):
        return "openai_responses", base_url
    if base_url.endswith("/messages"):
        return "anthropic_messages", base_url
    if "/v1beta/models" in base_url:
        if base_url.endswith(":generateContent"):
            return "gemini", base_url
        if "/v1beta/models/" in base_url:
            return "gemini", f"{base_url}:generateContent"
        return "gemini", f"{base_url.rstrip('/')}/{model_id}:generateContent"
    if base_url.endswith("/v1beta"):
        return "gemini", f"{base_url}/models/{model_id}:generateContent"
    if model_lower.startswith(("gemini", "models/gemini")):
        return "gemini", f"{base_url}/v1beta/models/{model_id}:generateContent"
    if model_lower.startswith(("claude", "anthropic")):
        if base_url.endswith("/v1"):
            return "anthropic_messages", f"{base_url}/messages"
        return "anthropic_messages", f"{base_url}/v1/messages"
    if base_url.endswith("/v1"):
        return "openai_chat", f"{base_url}/chat/completions"
    return "openai_chat", f"{base_url}/v1/chat/completions"


def protocol_choice_to_code(choice):
    return {
        "OpenAI Chat": "openai_chat",
        "OpenAI Responses": "openai_responses",
        "Gemini 原生": "gemini",
        "Claude Messages": "anthropic_messages",
    }.get(choice)


def resolve_protocol_url(base_url, model_id, protocol_code, empty_message, invalid_message):
    """Build a concrete endpoint from root URLs, version URLs, or full endpoints."""
    base_url = normalize_base_url(base_url, empty_message, invalid_message)
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("请填写模型 ID。")

    if protocol_code == "openai_chat":
        if base_url.endswith("/chat/completions"):
            return protocol_code, base_url
        if base_url.endswith("/v1"):
            return protocol_code, f"{base_url}/chat/completions"
        return protocol_code, f"{base_url}/v1/chat/completions"
    if protocol_code == "openai_responses":
        if base_url.endswith("/responses"):
            return protocol_code, base_url
        if base_url.endswith("/v1"):
            return protocol_code, f"{base_url}/responses"
        return protocol_code, f"{base_url}/v1/responses"
    if protocol_code == "anthropic_messages":
        if base_url.endswith("/messages"):
            return protocol_code, base_url
        if base_url.endswith("/v1"):
            return protocol_code, f"{base_url}/messages"
        return protocol_code, f"{base_url}/v1/messages"
    if protocol_code == "gemini":
        if base_url.endswith(":generateContent"):
            return protocol_code, base_url
        if "/v1beta/models/" in base_url:
            return protocol_code, f"{base_url}:generateContent"
        if "/v1beta/models" in base_url:
            return protocol_code, f"{base_url.rstrip('/')}/{model_id}:generateContent"
        if base_url.endswith("/v1beta"):
            return protocol_code, f"{base_url}/models/{model_id}:generateContent"
        return protocol_code, f"{base_url}/v1beta/models/{model_id}:generateContent"
    raise ValueError("不支持的协议选择。")


def resolve_text_protocol(base_url, model_id, protocol_choice):
    protocol_code = protocol_choice_to_code(protocol_choice)
    if protocol_code:
        return resolve_protocol_url(
            base_url,
            model_id,
            protocol_code,
            "请填写文本模型 API 地址。",
            "文本模型 API 地址格式不正确。",
        )
    return detect_text_protocol(base_url, model_id)


def detect_vision_protocol(base_url, model_id):
    base_url = normalize_base_url(base_url, "请填写多模态模型 API 地址。", "多模态模型 API 地址格式不正确。")
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("请填写多模态模型 ID。")

    model_lower = model_id.lower()
    if base_url.endswith("/chat/completions"):
        return "openai_chat", base_url
    if base_url.endswith("/responses"):
        return "openai_responses", base_url
    if base_url.endswith("/messages"):
        return "anthropic_messages", base_url
    if "/v1beta/models" in base_url:
        if base_url.endswith(":generateContent"):
            return "gemini", base_url
        if "/v1beta/models/" in base_url:
            return "gemini", f"{base_url}:generateContent"
        return "gemini", f"{base_url.rstrip('/')}/{model_id}:generateContent"
    if base_url.endswith("/v1beta"):
        return "gemini", f"{base_url}/models/{model_id}:generateContent"
    if model_lower.startswith(("claude", "anthropic")):
        if base_url.endswith("/v1"):
            return "anthropic_messages", f"{base_url}/messages"
        return "anthropic_messages", f"{base_url}/v1/messages"
    if model_lower.startswith(("gpt", "o1", "o3", "o4")):
        return "openai_chat", f"{base_url}/v1/chat/completions"
    if base_url.endswith("/v1"):
        return "openai_chat", f"{base_url}/chat/completions"
    return "gemini", f"{base_url}/v1beta/models/{model_id}:generateContent"


def resolve_vision_protocol(base_url, model_id, protocol_choice):
    protocol_code = protocol_choice_to_code(protocol_choice)
    if protocol_code:
        return resolve_protocol_url(
            base_url,
            model_id,
            protocol_code,
            "请填写多模态模型 API 地址。",
            "多模态模型 API 地址格式不正确。",
        )
    return detect_vision_protocol(base_url, model_id)


def get_save_dir(save_dir):
    save_dir = (save_dir or "").strip()
    if not save_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, "AI_Cards")
    return save_dir


def build_gallery_items(saved_paths):
    return [(path, f"第 {index} 张") for index, path in enumerate(saved_paths, start=1)]


def get_image_dimensions(image_path):
    """Read PNG/JPEG dimensions without pulling in another imaging dependency."""
    try:
        with open(image_path, "rb") as image_file:
            header = image_file.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width = int.from_bytes(header[16:20], "big")
                height = int.from_bytes(header[20:24], "big")
                return f"{width}x{height}"
            if header[:3] == b"\xff\xd8\xff":
                image_file.seek(2)
                while True:
                    marker_prefix = image_file.read(1)
                    if marker_prefix != b"\xff":
                        return ""
                    marker = image_file.read(1)
                    while marker == b"\xff":
                        marker = image_file.read(1)
                    if marker in (b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"):
                        segment_length = int.from_bytes(image_file.read(2), "big")
                        segment = image_file.read(segment_length - 2)
                        if len(segment) >= 5:
                            height = int.from_bytes(segment[1:3], "big")
                            width = int.from_bytes(segment[3:5], "big")
                            return f"{width}x{height}"
                    else:
                        segment_length_data = image_file.read(2)
                        if len(segment_length_data) != 2:
                            return ""
                        segment_length = int.from_bytes(segment_length_data, "big")
                        image_file.seek(segment_length - 2, os.SEEK_CUR)
    except Exception:
        return ""
    return ""


def format_resolution_summary(image_records, fallback_size):
    if not image_records:
        return f"图片分辨率：未保存图片；请求尺寸 {fallback_size}"

    dimensions_by_index = []
    for job_index, image_path, _elapsed in sorted(image_records, key=lambda item: item[0]):
        dimensions_by_index.append((job_index, get_image_dimensions(image_path) or fallback_size))

    unique_dimensions = {dimensions for _index, dimensions in dimensions_by_index}
    if len(unique_dimensions) == 1 and len(dimensions_by_index) > 12:
        only_dimensions = next(iter(unique_dimensions))
        return f"图片分辨率：第 {dimensions_by_index[0][0]}-{dimensions_by_index[-1][0]} 张均为 {only_dimensions}"

    details = "，".join(f"第 {index} 张 {dimensions}" for index, dimensions in dimensions_by_index)
    return f"图片分辨率：{details}"


def format_generation_stats(image_records, requested_count, total_elapsed, fallback_size):
    success_count = len(image_records)
    success_rate = success_count / requested_count * 100 if requested_count else 0
    average_elapsed = sum(record[2] for record in image_records) / success_count if success_count else 0
    return (
        f"{format_resolution_summary(image_records, fallback_size)}；"
        f"单张平均耗时 {format_duration(average_elapsed)}；"
        f"成功率 {success_count}/{requested_count} ({success_rate:.1f}%)；"
        f"总耗时 {format_duration(total_elapsed)}"
    )


def format_failed_jobs_summary(failed_jobs, max_items=3):
    if not failed_jobs:
        return ""
    samples = "；".join(
        f"第 {job_index} 张：{message[:260]}"
        for job_index, message in failed_jobs[-max_items:]
    )
    extra_count = len(failed_jobs) - max_items
    suffix = f"；另有 {extra_count} 个失败未展开" if extra_count > 0 else ""
    reconnect_hint = ""
    if any(is_remote_disconnected_error(message) for _job_index, message in failed_jobs):
        reconnect_hint = "\n提示：远端在返回结果前断开连接，API可能已生成并计费，但本地应用没有拿到图片 URL/base64，无法自动保存。可能是提示词被上游审核拦截或者上游服务异常。"
    return f"\n失败详情：{samples}{suffix}{reconnect_hint}"


def is_remote_disconnected_error(error):
    error_text = f"{type(error).__name__}: {error!r} {error}"
    return (
        "RemoteDisconnected" in error_text
        or "Remote end closed connection without response" in error_text
        or "Connection aborted" in error_text
    )


def extract_error_message(payload):
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return error.get("message") or error.get("code") or str(error)
        if isinstance(error, str):
            return error
        for key in ("message", "msg", "detail"):
            if payload.get(key):
                return str(payload[key])
    return ""


def format_response_error(response):
    try:
        payload = response.json()
        detail = extract_error_message(payload) or str(payload)
    except ValueError:
        detail = response.text.strip()

    if response.status_code in (401, 403):
        prefix = "认证失败，请检查 API Key 是否正确"
    elif response.status_code == 404:
        prefix = "接口地址不可用，请检查 API 地址或模型是否正确"
    elif response.status_code == 429:
        prefix = "请求过于频繁或额度不足"
    else:
        prefix = "中转站返回错误"

    detail = detail[:500] if detail else "没有返回错误详情"
    return f"{prefix}；HTTP {response.status_code}；{detail}"


def parse_image_items(response):
    if not response.ok:
        raise RuntimeError(format_response_error(response))

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError("中转站返回的不是有效 JSON。") from error

    error_message = extract_error_message(payload)
    if error_message:
        raise RuntimeError(f"中转站返回错误：{error_message}")

    if not isinstance(payload, dict):
        raise RuntimeError("中转站返回格式不正确：顶层不是 JSON 对象。")

    image_items = payload.get("data", [])
    if not isinstance(image_items, list):
        raise RuntimeError("中转站返回格式不正确：data 不是列表。")

    return image_items


def post_json(url, headers, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        **headers,
    }
    return requests.post(url, headers=request_headers, data=body, timeout=timeout)


def parse_text_model_content(protocol, response):
    if protocol == "gemini":
        return parse_google_content(response)

    if not response.ok:
        raise RuntimeError(format_response_error(response))

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError("文本模型接口返回的不是有效 JSON。") from error

    error_message = extract_error_message(payload)
    if error_message:
        raise RuntimeError(f"文本模型接口返回错误：{error_message}")

    if protocol == "openai_responses":
        output_text = payload.get("output_text")
        if output_text:
            return output_text.strip()
        texts = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text") and content.get("text"):
                    texts.append(content["text"])
        content = "\n".join(texts).strip()
        if content:
            return content

    if protocol == "anthropic_messages":
        texts = [part.get("text", "") for part in payload.get("content", []) if isinstance(part, dict)]
        content = "\n".join(text.strip() for text in texts if text.strip()).strip()
        if content:
            return content

    choices = payload.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if content:
            return content.strip()

    raise RuntimeError("文本模型接口返回内容为空。")


def parse_google_content(response):
    if not response.ok:
        raise RuntimeError(format_response_error(response))

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError("视觉评估接口返回的不是有效 JSON。") from error

    error_message = extract_error_message(payload)
    if error_message:
        raise RuntimeError(f"视觉评估接口返回错误：{error_message}")

    candidates = payload.get("candidates", [])
    if not candidates:
        raise RuntimeError("视觉评估接口没有返回候选结果。")

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    content = "\n".join(text.strip() for text in texts if text.strip()).strip()
    if not content:
        raise RuntimeError("视觉评估接口返回内容为空。")
    return content


def prepare_vision_image(image_path):
    """Create a compact JPEG copy in memory for multimodal evaluation."""
    original_size = os.path.getsize(image_path)
    with Image.open(image_path) as image:
        image = image.convert("RGBA")
        if max(image.size) > VISION_IMAGE_MAX_SIDE:
            image.thumbnail((VISION_IMAGE_MAX_SIDE, VISION_IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=VISION_IMAGE_JPEG_QUALITY,
            optimize=True,
            progressive=True,
        )

    data = buffer.getvalue()
    return {
        "base64": base64.b64encode(data).decode("utf-8"),
        "mime_type": "image/jpeg",
        "original_size": original_size,
        "compressed_size": len(data),
        "dimensions": f"{image.width}x{image.height}",
    }


def optimize_prompt_with_image(
    prompt,
    image_path,
    base_url,
    model_id,
    api_key,
    protocol_choice="自动识别",
    retry_count=1,
    retry_delay=2,
    on_retry=None,
    reasoning_effort="关闭",
    creation_theme="",
    user_initial_direction="",
    prepared_vision_image=None,
):
    if not api_key or not api_key.strip():
        raise ValueError("请填写视觉评估 API Key。")

    vision_image = prepared_vision_image or prepare_vision_image(image_path)
    image_base64 = vision_image["base64"]
    image_mime_type = vision_image["mime_type"]
    optimizer_prompt = CONFIG.get("iteration_optimizer_prompt", ITERATION_OPTIMIZER_PROMPT)
    context_parts = [
        f"【多模态模型的系统提示词】\n{optimizer_prompt.strip()}",
    ]
    if creation_theme and creation_theme.strip():
        context_parts.append(f"【创作主题】\n{creation_theme.strip()}")
    if user_initial_direction and user_initial_direction.strip():
        context_parts.append(f"【用户初始创作方向】\n{user_initial_direction.strip()}")
    context_parts.append(f"【本轮图片使用的提示词】\n{prompt.strip()}")
    context_parts.append("【本轮图片】\n见随附图片。")
    request_text = "\n\n".join(context_parts)
    protocol, request_url = resolve_vision_protocol(base_url, model_id, protocol_choice)

    def request_vision():
        if protocol == "openai_chat":
            payload = apply_reasoning_settings(
                {
                    "model": model_id.strip(),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": request_text},
                                {"type": "image_url", "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"}},
                            ],
                        }
                    ],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
                json=payload,
                timeout=request_timeout(VISION_READ_TIMEOUT),
            )
        elif protocol == "openai_responses":
            payload = apply_reasoning_settings(
                {
                    "model": model_id.strip(),
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": request_text},
                                {"type": "input_image", "image_url": f"data:{image_mime_type};base64,{image_base64}"},
                            ],
                        }
                    ],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
                json=payload,
                timeout=request_timeout(VISION_READ_TIMEOUT),
            )
        elif protocol == "anthropic_messages":
            payload = apply_reasoning_settings(
                {
                    "model": model_id.strip(),
                    "max_tokens": 2000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": request_text},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image_mime_type,
                                        "data": image_base64,
                                    },
                                },
                            ],
                        }
                    ],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key.strip(),
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
                timeout=request_timeout(VISION_READ_TIMEOUT),
            )
        else:
            payload = apply_reasoning_settings(
                {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": request_text},
                                {
                                    "inline_data": {
                                        "mime_type": image_mime_type,
                                        "data": image_base64,
                                    }
                                },
                            ],
                        }
                    ],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers=gemini_headers(api_key, request_url),
                json=payload,
                timeout=request_timeout(VISION_READ_TIMEOUT),
            )
        return parse_text_model_content(protocol, response)

    return run_with_retry(
        request_vision,
        "视觉评估优化",
        retries=int(retry_count),
        delay_seconds=float(retry_delay),
        on_retry=on_retry,
    )


def request_multimodal_text(
    text_prompt,
    image_path,
    base_url,
    model_id,
    api_key,
    protocol_choice="自动识别",
    retry_count=1,
    retry_delay=2,
    reasoning_effort="关闭",
    on_retry=None,
    label="多模态请求",
):
    if not api_key or not api_key.strip():
        raise ValueError("请填写多模态模型 API Key。")
    if not model_id or not model_id.strip():
        raise ValueError("请填写多模态模型 ID。")

    vision_image = prepare_vision_image(image_path)
    image_base64 = vision_image["base64"]
    image_mime_type = vision_image["mime_type"]
    protocol, request_url = resolve_vision_protocol(base_url, model_id, protocol_choice)

    def request_vision_text():
        if protocol == "openai_chat":
            payload = apply_reasoning_settings(
                {
                    "model": model_id.strip(),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"}},
                            ],
                        }
                    ],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
                json=payload,
                timeout=request_timeout(VISION_READ_TIMEOUT),
            )
        elif protocol == "openai_responses":
            payload = apply_reasoning_settings(
                {
                    "model": model_id.strip(),
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": text_prompt},
                                {"type": "input_image", "image_url": f"data:{image_mime_type};base64,{image_base64}"},
                            ],
                        }
                    ],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
                json=payload,
                timeout=request_timeout(VISION_READ_TIMEOUT),
            )
        elif protocol == "anthropic_messages":
            payload = apply_reasoning_settings(
                {
                    "model": model_id.strip(),
                    "max_tokens": 2000,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_prompt},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image_mime_type,
                                        "data": image_base64,
                                    },
                                },
                            ],
                        }
                    ],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key.strip(),
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
                timeout=request_timeout(VISION_READ_TIMEOUT),
            )
        else:
            payload = apply_reasoning_settings(
                {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": text_prompt},
                                {
                                    "inline_data": {
                                        "mime_type": image_mime_type,
                                        "data": image_base64,
                                    }
                                },
                            ],
                        }
                    ],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers=gemini_headers(api_key, request_url),
                json=payload,
                timeout=request_timeout(VISION_READ_TIMEOUT),
            )
        return parse_text_model_content(protocol, response)

    content = run_with_retry(
        request_vision_text,
        label,
        retries=int(retry_count),
        delay_seconds=float(retry_delay),
        on_retry=on_retry,
    )
    return content, vision_image, protocol, request_url


def build_random_user_prompt(preference):
    preference = (preference or "").strip()
    user_prompt = CONFIG.get("random_user_prompt", RANDOM_USER_PROMPT)
    if not preference:
        return user_prompt
    return f"{user_prompt}\n\n本次创作方向：{preference}"


def resolve_reverse_image_path(uploaded_image, local_image_path):
    if uploaded_image:
        return uploaded_image
    local_image_path = (local_image_path or "").strip().strip('"')
    if not local_image_path:
        raise ValueError("请上传图片，或填写本地图片路径。")
    if not os.path.exists(local_image_path):
        raise ValueError("本地图片路径不存在。")
    return local_image_path


def reverse_prompt_from_image(
    uploaded_image,
    local_image_path,
    iteration_base_url,
    iteration_model_id,
    iteration_api_key,
    iteration_protocol,
    iteration_reasoning_effort,
    retry_count,
    retry_delay,
):
    reset_stop_flag("reverse")
    retry_count, retry_delay = normalize_retry_settings(retry_count, retry_delay)
    iteration_protocol = normalize_protocol(iteration_protocol)
    iteration_reasoning_effort = normalize_reasoning_effort(iteration_reasoning_effort)
    image_path = resolve_reverse_image_path(uploaded_image, local_image_path)
    reverse_prompt = CONFIG.get("reverse_prompt", REVERSE_PROMPT)
    started_at = time.perf_counter()
    retry_events = []

    try:
        result, vision_image, protocol, request_url = request_multimodal_text(
            reverse_prompt,
            image_path,
            iteration_base_url,
            iteration_model_id,
            iteration_api_key,
            iteration_protocol,
            retry_count,
            retry_delay,
            iteration_reasoning_effort,
            lambda label, attempt, retries, error: retry_events.append(
                f"{label}触发重试 {attempt}/{retries}：{error}"
            ),
            "提示词反推",
        )
    except Exception as e:
        return "", f"提示词反推失败：{e}"

    status_extra = f"\n{retry_events[-1]}" if retry_events else ""
    return (
        result,
        f"反推完成；协议：{format_protocol_label(protocol)}；地址：{display_endpoint(request_url)}；上传图片 {format_bytes(vision_image['original_size'])} -> {format_bytes(vision_image['compressed_size'])}，{vision_image['dimensions']}；耗时 {format_duration(time.perf_counter() - started_at)}{status_extra}",
    )


def generate_random_prompt(
    base_url,
    model_id,
    api_key,
    preference,
    protocol_choice="自动识别",
    retry_count=1,
    retry_delay=2,
    on_retry=None,
    reasoning_effort="关闭",
):
    if not api_key or not api_key.strip():
        raise ValueError("请填写随机提示词 API Key。")
    if not model_id or not model_id.strip():
        raise ValueError("请填写随机提示词模型 ID。")

    protocol, request_url = resolve_text_protocol(base_url, model_id, protocol_choice)
    system_prompt = CONFIG.get("random_system_prompt", RANDOM_SYSTEM_PROMPT)
    user_prompt = build_random_user_prompt(preference)

    def request_text():
        if protocol == "openai_chat":
            payload = apply_reasoning_settings(
                {
                    "model": model_id.strip(),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
                json=payload,
                timeout=request_timeout(TEXT_READ_TIMEOUT),
            )
        elif protocol == "openai_responses":
            payload = apply_reasoning_settings(
                {
                    "model": model_id.strip(),
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
                json=payload,
                timeout=request_timeout(TEXT_READ_TIMEOUT),
            )
        elif protocol == "anthropic_messages":
            payload = apply_reasoning_settings(
                {
                    "model": model_id.strip(),
                    "system": system_prompt,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key.strip(),
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
                timeout=request_timeout(TEXT_READ_TIMEOUT),
            )
        else:
            payload = apply_reasoning_settings(
                {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                },
                protocol,
                reasoning_effort,
                model_id,
            )
            response = requests.post(
                request_url,
                headers=gemini_headers(api_key, request_url),
                json=payload,
                timeout=request_timeout(TEXT_READ_TIMEOUT),
            )
        return parse_text_model_content(protocol, response)

    return run_with_retry(
        request_text,
        "随机提示词生成",
        retries=int(retry_count),
        delay_seconds=float(retry_delay),
        on_retry=on_retry,
    )


def is_http_url(value):
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def save_image_bytes(image_bytes, saved_paths, save_dir, timestamp):
    image_index = len(saved_paths) + 1
    filename = f"img_{timestamp}_{image_index:02d}.png"
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "wb") as f:
        f.write(image_bytes)
    saved_paths.append(save_path)


def save_image_url(image_url, saved_paths, save_dir, timestamp, retry_count=1, retry_delay=2, on_retry=None):
    def download_image():
        image_response = requests.get(image_url, timeout=request_timeout(IMAGE_READ_TIMEOUT))
        image_response.raise_for_status()
        return image_response.content

    image_bytes = run_with_retry(
        download_image,
        "图片下载",
        retries=int(retry_count),
        delay_seconds=float(retry_delay),
        on_retry=on_retry,
    )
    save_image_bytes(image_bytes, saved_paths, save_dir, timestamp)


def save_image_value(image_value, saved_paths, save_dir, timestamp, retry_count=1, retry_delay=2, on_retry=None):
    if not image_value:
        return

    if is_http_url(image_value):
        save_image_url(image_value, saved_paths, save_dir, timestamp, retry_count, retry_delay, on_retry)
        return

    image_bytes = base64.b64decode(image_value)
    save_image_bytes(image_bytes, saved_paths, save_dir, timestamp)


def save_images_from_items(image_items, saved_paths, save_dir, timestamp, retry_count=1, retry_delay=2, on_retry=None):
    for item in image_items:
        if isinstance(item, str):
            save_image_value(item, saved_paths, save_dir, timestamp, retry_count, retry_delay, on_retry)
            continue

        image_values = item.get("b64_json")
        if isinstance(image_values, str):
            image_values = [image_values]

        for image_base64 in image_values or []:
            save_image_value(image_base64, saved_paths, save_dir, timestamp, retry_count, retry_delay, on_retry)

    for item in image_items:
        if not isinstance(item, dict):
            continue

        image_url = item.get("url")
        if not image_url:
            continue

        save_image_url(image_url, saved_paths, save_dir, timestamp, retry_count, retry_delay, on_retry)


def generate_one_image(
    prompt,
    save_dir,
    aspect_ratio,
    resolution,
    base_url,
    model_id,
    quality,
    api_key,
    timestamp,
    retry_count=1,
    retry_delay=2,
    on_retry=None,
):
    size = resolve_size(aspect_ratio, resolution)
    quality = normalize_quality(quality)
    saved_paths = []
    image_items = run_with_retry(
        lambda: parse_image_items(
            post_json(
                resolve_api_url(base_url),
                headers={
                    "Authorization": f"Bearer {api_key.strip()}",
                },
                payload={
                    "model": model_id.strip(),
                    "size": size,
                    "n": 1,
                    "quality": quality,
                    "moderation": "low",
                    "prompt": prompt.strip(),
                },
                timeout=request_timeout(IMAGE_READ_TIMEOUT),
            )
        ),
        "图片生成",
        retries=int(retry_count),
        delay_seconds=float(retry_delay),
        on_retry=on_retry,
    )
    save_images_from_items(image_items, saved_paths, save_dir, timestamp, retry_count, retry_delay, on_retry)
    if not saved_paths:
        raise RuntimeError("接口返回成功，但没有收到图片数据。")
    return saved_paths[0]


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    minutes = int(seconds // 60)
    rest_seconds = seconds % 60
    return f"{minutes} 分 {rest_seconds:.1f} 秒"


def run_with_retry(action, label, retries=1, delay_seconds=2, on_retry=None):
    """Retry a single API/download operation; callers decide whether a failed job is skipped."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return action()
        except Exception as error:
            last_error = error
            if label == "图片生成" and is_remote_disconnected_error(error):
                raise RuntimeError(
                    f"{label}遇到 RemoteDisconnected，不再自动重试，避免 API 可能已生成并计费后重复请求：{error}"
                ) from error
            if attempt < retries:
                if on_retry:
                    on_retry(label, attempt + 1, retries, error)
                time.sleep(delay_seconds)
    raise RuntimeError(f"{label}失败，已自动重试 {retries} 次：{last_error}") from last_error


def generate_images_concurrently(
    prompt_jobs,
    save_dir,
    aspect_ratio,
    resolution,
    base_url,
    model_id,
    quality,
    api_key,
    concurrency,
    retry_count=1,
    retry_delay=2,
    stop_mode=None,
):
    save_dir = get_save_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    prompt_jobs = list(prompt_jobs)
    total_count = len(prompt_jobs)
    concurrency = max(1, min(int(concurrency), total_count or 1))
    saved_paths = []
    image_records = []
    failed_jobs = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    total_started_at = time.perf_counter()
    request_size = resolve_size(aspect_ratio, resolution)
    quality = normalize_quality(quality)

    if not prompt_jobs:
        yield [], "没有可生成的提示词。"
        return

    yield (
        build_gallery_items(saved_paths),
        f"开始生成 {total_count} 张；最大并发 {concurrency}；请求尺寸 {request_size}；品质 {quality}",
    )

    def worker(job):
        job_index, job_prompt = job
        if stop_mode and should_stop(stop_mode):
            raise RuntimeError("任务已停止。")
        started_at = time.perf_counter()
        events = []
        image_path = generate_one_image(
            job_prompt,
            save_dir,
            aspect_ratio,
            resolution,
            base_url,
            model_id,
            quality,
            api_key,
            f"{timestamp}_job{job_index:02d}",
            retry_count,
            retry_delay,
            lambda label, attempt, retries, error: events.append(
                f"第 {job_index} 张{label}触发重试 {attempt}/{retries}：{error}"
            ),
        )
        return job_index, job_prompt, image_path, time.perf_counter() - started_at, events

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Keep only a small window of futures queued so Stop can prevent later jobs from starting.
            job_iter = iter(prompt_jobs)
            future_to_index = {}

            def submit_next_job():
                try:
                    job = next(job_iter)
                except StopIteration:
                    return
                future_to_index[executor.submit(worker, job)] = job[0]

            for _ in range(concurrency):
                submit_next_job()

            while future_to_index:
                if stop_mode and should_stop(stop_mode):
                    for pending_future in future_to_index:
                        pending_future.cancel()
                    yield (
                        build_gallery_items(saved_paths),
                        f"已停止：保存 {len(saved_paths)}/{total_count} 张。",
                    )
                    return
                done, _pending = wait(set(future_to_index), return_when=FIRST_COMPLETED)
                future = next(iter(done))
                try:
                    job_index, _job_prompt, image_path, elapsed, events = future.result()
                    saved_paths.append(image_path)
                    image_records.append((job_index, image_path, elapsed))
                    dimensions = get_image_dimensions(image_path) or request_size
                    status_extra = f"\n{events[-1]}" if events else ""
                    yield (
                        build_gallery_items(saved_paths),
                        f"已完成 {len(saved_paths)}/{total_count} 张；刚完成第 {job_index} 张，分辨率 {dimensions}，耗时 {format_duration(elapsed)}；累计耗时 {format_duration(time.perf_counter() - total_started_at)}{status_extra}",
                    )
                except Exception as e:
                    job_index = future_to_index[future]
                    failed_jobs.append((job_index, str(e)))
                    yield (
                        build_gallery_items(saved_paths),
                        f"第 {job_index} 张失败并已跳过：{e}；已保存 {len(saved_paths)}/{total_count} 张，失败 {len(failed_jobs)} 张。",
                    )
                finally:
                    future_to_index.pop(future, None)
                    if not (stop_mode and should_stop(stop_mode)):
                        submit_next_job()
    except Exception as e:
        failed_count = len(failed_jobs) or total_count - len(saved_paths)
        yield (
            build_gallery_items(saved_paths),
            f"生成中断：{e}；已保存 {len(saved_paths)}/{total_count} 张，失败 {failed_count} 张；{format_generation_stats(image_records, total_count, time.perf_counter() - total_started_at, request_size)}。",
        )
        return

    failed_summary = f"；失败 {len(failed_jobs)} 张" if failed_jobs else ""
    yield (
        build_gallery_items(saved_paths),
        f"生成完成：共保存 {len(saved_paths)} 张{failed_summary}；{format_generation_stats(image_records, total_count, time.perf_counter() - total_started_at, request_size)}；品质 {quality}；目录 {save_dir}{format_failed_jobs_summary(failed_jobs)}",
    )


def generate_images_from_prompt(
    prompt,
    save_dir,
    image_count,
    aspect_ratio,
    resolution,
    base_url,
    model_id,
    quality,
    api_key,
    concurrency=1,
    retry_count=1,
    retry_delay=2,
    stop_mode=None,
):
    prompt_jobs = [(index, prompt) for index in range(1, int(image_count) + 1)]
    yield from generate_images_concurrently(
        prompt_jobs,
        save_dir,
        aspect_ratio,
        resolution,
        base_url,
        model_id,
        quality,
        api_key,
        concurrency,
        retry_count,
        retry_delay,
        stop_mode,
    )


def generate_image(prompt, save_dir, image_count, concurrency, retry_count, retry_delay, aspect_ratio, resolution, base_url, model_id, quality, api_key):
    reset_stop_flag("manual")
    if not prompt or not prompt.strip():
        yield [], "请先输入提示词。"
        return
    if not api_key or not api_key.strip():
        yield [], "请填写 API Key。"
        return
    if not model_id or not model_id.strip():
        yield [], "请填写模型 ID。"
        return

    retry_count, retry_delay = normalize_retry_settings(retry_count, retry_delay)
    quality = normalize_quality(quality)

    persist_config(
        {
            "prompt": prompt,
            "save_dir": save_dir,
            "image_count": int(image_count),
            "concurrency": int(concurrency),
            "retry_count": int(retry_count),
            "retry_delay": float(retry_delay),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "base_url": base_url,
            "model_id": model_id,
            "quality": quality,
            "api_key": api_key,
        },
    )

    yield from generate_images_from_prompt(
        prompt,
        save_dir,
        image_count,
        aspect_ratio,
        resolution,
        base_url,
        model_id,
        quality,
        api_key,
        concurrency,
        retry_count,
        retry_delay,
        "manual",
    )


def generate_random_image(
    save_dir,
    image_count,
    concurrency,
    retry_count,
    retry_delay,
    aspect_ratio,
    resolution,
    base_url,
    model_id,
    quality,
    api_key,
    random_base_url,
    random_model_id,
    random_api_key,
    random_protocol,
    random_reasoning_effort,
    random_preference,
):
    reset_stop_flag("random")
    if not api_key or not api_key.strip():
        yield "", [], "请填写图片生成 API Key。"
        return
    if not model_id or not model_id.strip():
        yield "", [], "请填写图片生成模型 ID。"
        return

    retry_count, retry_delay = normalize_retry_settings(retry_count, retry_delay)
    quality = normalize_quality(quality)
    random_protocol = normalize_protocol(random_protocol)
    random_reasoning_effort = normalize_reasoning_effort(random_reasoning_effort)
    prompt_started_at = time.perf_counter()
    persist_config(
        {
            "save_dir": save_dir,
            "image_count": int(image_count),
            "concurrency": int(concurrency),
            "retry_count": int(retry_count),
            "retry_delay": float(retry_delay),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "base_url": base_url,
            "model_id": model_id,
            "quality": quality,
            "api_key": api_key,
            "random_base_url": random_base_url,
            "random_model_id": random_model_id,
            "random_api_key": random_api_key,
            "random_protocol": random_protocol,
            "random_reasoning_effort": random_reasoning_effort,
            "random_preference": random_preference,
        },
    )
    yield "", [], "正在生成随机提示词..."
    if should_stop("random"):
        yield "", [], "已停止。"
        return

    try:
        random_prompt = generate_random_prompt(
            random_base_url,
            random_model_id,
            random_api_key,
            random_preference,
            random_protocol,
            retry_count,
            retry_delay,
            reasoning_effort=random_reasoning_effort,
        )
    except Exception as e:
        yield "", [], f"随机提示词生成失败：{e}"
        return

    persist_config({"prompt": random_prompt})

    yield (
        random_prompt,
        [],
        f"随机提示词已生成；耗时 {format_duration(time.perf_counter() - prompt_started_at)}。开始生成图片...",
    )

    for gallery_items, status in generate_images_from_prompt(
        random_prompt,
        save_dir,
        image_count,
        aspect_ratio,
        resolution,
        base_url,
        model_id,
        quality,
        api_key,
        concurrency,
        retry_count,
        retry_delay,
        "random",
    ):
        yield random_prompt, gallery_items, status


def generate_creative_images(
    save_dir,
    creative_count,
    text_concurrency,
    image_concurrency,
    retry_count,
    retry_delay,
    aspect_ratio,
    resolution,
    image_base_url,
    image_model_id,
    quality,
    image_api_key,
    random_base_url,
    random_model_id,
    random_api_key,
    random_protocol,
    random_reasoning_effort,
    random_preference,
):
    reset_stop_flag("creative")
    if not image_api_key or not image_api_key.strip():
        yield "", [], "请填写图片生成 API Key。"
        return
    if not image_model_id or not image_model_id.strip():
        yield "", [], "请填写图片生成模型 ID。"
        return

    creative_count = int(creative_count)
    text_concurrency = max(1, int(text_concurrency))
    image_concurrency = max(1, int(image_concurrency))
    retry_count, retry_delay = normalize_retry_settings(retry_count, retry_delay)
    total_started_at = time.perf_counter()
    prompts = []
    saved_paths = []
    image_records = []
    save_dir = get_save_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    request_size = resolve_size(aspect_ratio, resolution)
    quality = normalize_quality(quality)
    random_protocol = normalize_protocol(random_protocol)
    random_reasoning_effort = normalize_reasoning_effort(random_reasoning_effort)

    persist_config(
        {
            "save_dir": save_dir,
            "creative_count": creative_count,
            "text_concurrency": text_concurrency,
            "image_concurrency": image_concurrency,
            "retry_count": retry_count,
            "retry_delay": retry_delay,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "base_url": image_base_url,
            "model_id": image_model_id,
            "quality": quality,
            "api_key": image_api_key,
            "random_base_url": random_base_url,
            "random_model_id": random_model_id,
            "random_api_key": random_api_key,
            "random_protocol": random_protocol,
            "random_reasoning_effort": random_reasoning_effort,
            "random_preference": random_preference,
        },
    )

    retry_events = []
    failed_prompts = []
    failed_images = []
    yield "", [], f"正在生成 {creative_count} 段随机提示词并流水线出图；文本并发 {text_concurrency}，图片并发 {image_concurrency}；请求尺寸 {request_size}；品质 {quality}。"

    def prompt_worker(index):
        events = []
        prompt = generate_random_prompt(
            random_base_url,
            random_model_id,
            random_api_key,
            random_preference,
            random_protocol,
            retry_count,
            retry_delay,
            lambda label, attempt, retries, error: events.append(
                f"第 {index} 段提示词触发重试 {attempt}/{retries}：{error}"
            ),
            reasoning_effort=random_reasoning_effort,
        )
        return index, prompt, events

    def image_worker(index, prompt):
        started_at = time.perf_counter()
        events = []
        image_path = generate_one_image(
            prompt,
            save_dir,
            aspect_ratio,
            resolution,
            image_base_url,
            image_model_id,
            quality,
            image_api_key,
            f"{timestamp}_creative{index:02d}",
            retry_count,
            retry_delay,
            lambda label, attempt, retries, error: events.append(
                f"第 {index} 张图片触发重试 {attempt}/{retries}：{error}"
            ),
        )
        return index, image_path, time.perf_counter() - started_at, events

    try:
        with ThreadPoolExecutor(max_workers=min(text_concurrency, creative_count)) as prompt_executor, ThreadPoolExecutor(
            max_workers=min(image_concurrency, creative_count)
        ) as image_executor:
            # Prompt generation and image generation run as a pipeline: each prompt starts its image job immediately.
            prompt_futures = {
                prompt_executor.submit(prompt_worker, index): index
                for index in range(1, creative_count + 1)
            }
            image_futures = {}
            pending_prompt_futures = set(prompt_futures)

            while pending_prompt_futures or image_futures:
                if should_stop("creative"):
                    for pending_future in pending_prompt_futures:
                        pending_future.cancel()
                    for pending_future in image_futures:
                        pending_future.cancel()
                    prompt_text = "\n\n".join(f"第 {i} 段提示词：\n{text}" for i, text in sorted(prompts))
                    yield (
                        prompt_text,
                        build_gallery_items(saved_paths),
                        f"已停止：已生成提示词 {len(prompts)}/{creative_count} 段，已保存图片 {len(saved_paths)}/{creative_count} 张。",
                    )
                    return
                pending_work = set(pending_prompt_futures) | set(image_futures)
                done, _pending = wait(pending_work, return_when=FIRST_COMPLETED)

                for future in done:
                    if future in pending_prompt_futures:
                        pending_prompt_futures.remove(future)
                        prompt_text = "\n\n".join(f"第 {i} 段提示词：\n{text}" for i, text in sorted(prompts))
                        try:
                            index, prompt, events = future.result()
                            retry_events.extend(events)
                            prompts.append((index, prompt))
                            image_futures[image_executor.submit(image_worker, index, prompt)] = index
                            prompt_text = "\n\n".join(f"第 {i} 段提示词：\n{text}" for i, text in sorted(prompts))
                            status_extra = f"\n{retry_events[-1]}" if retry_events else ""
                            yield (
                                prompt_text,
                                build_gallery_items(saved_paths),
                                f"已生成 {len(prompts)}/{creative_count} 段提示词；已启动第 {index} 张生图；已完成 {len(saved_paths)}/{creative_count} 张。{status_extra}",
                            )
                        except Exception as e:
                            index = prompt_futures[future]
                            failed_prompts.append((index, str(e)))
                            status_extra = f"\n跳过第 {index} 段提示词：{e}"
                            yield (
                                prompt_text,
                                build_gallery_items(saved_paths),
                                f"提示词失败 {len(failed_prompts)} 段；已完成图片 {len(saved_paths)}/{creative_count} 张。{status_extra}",
                            )
                    elif future in image_futures:
                        prompt_text = "\n\n".join(f"第 {i} 段提示词：\n{text}" for i, text in sorted(prompts))
                        index = image_futures.pop(future)
                        try:
                            index, image_path, elapsed, events = future.result()
                            retry_events.extend(events)
                            saved_paths.append(image_path)
                            image_records.append((index, image_path, elapsed))
                            dimensions = get_image_dimensions(image_path) or request_size
                            status_extra = f"\n{retry_events[-1]}" if events else ""
                            yield (
                                prompt_text,
                                build_gallery_items(saved_paths),
                                f"已完成 {len(saved_paths)}/{creative_count} 张；刚完成第 {index} 张，分辨率 {dimensions}，耗时 {format_duration(elapsed)}；累计耗时 {format_duration(time.perf_counter() - total_started_at)}{status_extra}",
                            )
                        except Exception as e:
                            failed_images.append((index, str(e)))
                            yield (
                                prompt_text,
                                build_gallery_items(saved_paths),
                                f"第 {index} 张图片失败并已跳过：{e}；已保存 {len(saved_paths)}/{creative_count} 张，失败 {len(failed_images)} 张。",
                            )

            if prompts:
                persist_config({"prompt": sorted(prompts)[-1][1]})

    except Exception as e:
        prompt_text = "\n\n".join(f"第 {i} 段提示词：\n{text}" for i, text in sorted(prompts))
        yield (
            prompt_text,
            build_gallery_items(saved_paths),
            f"创意模式调度中断：{e}；已生成提示词 {len(prompts)}/{creative_count} 段，已保存图片 {len(saved_paths)}/{creative_count} 张。",
        )
        return

    prompt_text = "\n\n".join(f"第 {i} 段提示词：\n{text}" for i, text in sorted(prompts))
    yield (
        prompt_text,
        build_gallery_items(saved_paths),
        f"创意模式完成：共生成 {len(prompts)} 段提示词，保存 {len(saved_paths)} 张图片，提示词失败 {len(failed_prompts)} 段，图片失败 {len(failed_images)} 张；{format_generation_stats(image_records, creative_count, time.perf_counter() - total_started_at, request_size)}；品质 {quality}；目录 {save_dir}",
    )


def generate_iterative_image(
    save_dir,
    iteration_prompt_source,
    iteration_custom_prompt,
    iteration_count,
    retry_count,
    retry_delay,
    aspect_ratio,
    resolution,
    image_base_url,
    image_model_id,
    quality,
    image_api_key,
    random_base_url,
    random_model_id,
    random_api_key,
    random_protocol,
    random_reasoning_effort,
    random_preference,
    iteration_base_url,
    iteration_model_id,
    iteration_api_key,
    iteration_protocol,
    iteration_reasoning_effort,
):
    reset_stop_flag("iterative")
    if not image_api_key or not image_api_key.strip():
        yield "", [], "请填写图片生成 API Key。"
        return
    if not image_model_id or not image_model_id.strip():
        yield "", [], "请填写图片生成模型 ID。"
        return

    iteration_prompt_source = normalize_iteration_prompt_source(iteration_prompt_source)
    iteration_custom_prompt = (iteration_custom_prompt or "").strip()
    iteration_count = int(iteration_count)
    retry_count, retry_delay = normalize_retry_settings(retry_count, retry_delay)
    quality = normalize_quality(quality)
    random_protocol = normalize_protocol(random_protocol)
    iteration_protocol = normalize_protocol(iteration_protocol)
    random_reasoning_effort = normalize_reasoning_effort(random_reasoning_effort)
    iteration_reasoning_effort = normalize_reasoning_effort(iteration_reasoning_effort)
    raw_save_dir = save_dir
    save_dir = get_save_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    total_started_at = time.perf_counter()
    saved_paths = []
    image_records = []
    prompt_history = []
    request_size = resolve_size(aspect_ratio, resolution)

    persist_config(
        {
            "save_dir": raw_save_dir,
            "iteration_count": iteration_count,
            "retry_count": retry_count,
            "retry_delay": retry_delay,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "base_url": image_base_url,
            "model_id": image_model_id,
            "quality": quality,
            "api_key": image_api_key,
            "random_base_url": random_base_url,
            "random_model_id": random_model_id,
            "random_api_key": random_api_key,
            "random_protocol": random_protocol,
            "random_reasoning_effort": random_reasoning_effort,
            "random_preference": random_preference,
            "iteration_prompt_source": iteration_prompt_source,
            "iteration_custom_prompt": iteration_custom_prompt,
            "iteration_base_url": iteration_base_url,
            "iteration_model_id": iteration_model_id,
            "iteration_api_key": iteration_api_key,
            "iteration_protocol": iteration_protocol,
            "iteration_reasoning_effort": iteration_reasoning_effort,
        },
    )

    if iteration_prompt_source == "自定义提示词":
        if not iteration_custom_prompt:
            yield "", [], "请填写自定义初始提示词。"
            return
        current_prompt = iteration_custom_prompt
        yield "", [], "已使用自定义初始提示词，开始第 1 轮出图..."
    else:
        yield "", [], "正在生成初始随机提示词..."
        try:
            current_prompt = generate_random_prompt(
                random_base_url,
                random_model_id,
                random_api_key,
                random_preference,
                random_protocol,
                retry_count,
                retry_delay,
                reasoning_effort=random_reasoning_effort,
            )
        except Exception as e:
            yield "", [], f"初始随机提示词生成失败：{e}"
            return

    prompt_history.append(f"第 1 轮提示词：\n{current_prompt}")
    persist_config({"prompt": current_prompt})
    yield "\n\n".join(prompt_history), [], "初始提示词已生成，开始第 1 轮出图..."

    for round_index in range(1, iteration_count + 1):
        if should_stop("iterative"):
            yield (
                "\n\n".join(prompt_history),
                build_gallery_items(saved_paths),
                f"已停止：已生成 {len(saved_paths)}/{iteration_count} 张。",
            )
            return
        round_started_at = time.perf_counter()
        try:
            yield (
                "\n\n".join(prompt_history),
                build_gallery_items(saved_paths),
                f"正在生成第 {round_index}/{iteration_count} 轮图片...",
            )
            image_path = generate_one_image(
                current_prompt,
                save_dir,
                aspect_ratio,
                resolution,
                image_base_url,
                image_model_id,
                quality,
                image_api_key,
                f"{timestamp}_round{round_index:02d}",
                retry_count,
                retry_delay,
            )
            saved_paths.append(image_path)
            elapsed = time.perf_counter() - round_started_at
            image_records.append((round_index, image_path, elapsed))
            dimensions = get_image_dimensions(image_path) or request_size
            yield (
                "\n\n".join(prompt_history),
                build_gallery_items(saved_paths),
                f"第 {round_index}/{iteration_count} 轮图片已生成；分辨率 {dimensions}；本轮耗时 {format_duration(elapsed)}",
            )

            if round_index >= iteration_count:
                break

            if should_stop("iterative"):
                yield (
                    "\n\n".join(prompt_history),
                    build_gallery_items(saved_paths),
                    f"已停止：已生成 {len(saved_paths)}/{iteration_count} 张。",
                )
                return

            vision_protocol, vision_url = resolve_vision_protocol(
                iteration_base_url,
                iteration_model_id,
                iteration_protocol,
            )
            preview_image = prepare_vision_image(image_path)
            upload_size_label = f"{format_bytes(preview_image['original_size'])} -> {format_bytes(preview_image['compressed_size'])}"
            yield (
                "\n\n".join(prompt_history),
                build_gallery_items(saved_paths),
                f"正在用多模态模型评估第 {round_index} 轮图片并优化提示词... 协议：{format_protocol_label(vision_protocol)}；地址：{display_endpoint(vision_url)}；上传图片 {upload_size_label}，{preview_image['dimensions']}",
            )
            current_prompt = optimize_prompt_with_image(
                current_prompt,
                image_path,
                iteration_base_url,
                iteration_model_id,
                iteration_api_key,
                iteration_protocol,
                retry_count,
                retry_delay,
                reasoning_effort=iteration_reasoning_effort,
                creation_theme=random_preference if iteration_prompt_source == "自定义提示词" else "",
                user_initial_direction=random_preference if iteration_prompt_source == "随机提示词" else "",
                prepared_vision_image=preview_image,
            )
            prompt_history.append(f"第 {round_index + 1} 轮提示词：\n{current_prompt}")
            persist_config({"prompt": current_prompt})
            yield (
                "\n\n".join(prompt_history),
                build_gallery_items(saved_paths),
                f"第 {round_index + 1} 轮提示词已优化完成。",
            )

        except requests.exceptions.RequestException as e:
            yield (
                "\n\n".join(prompt_history),
                build_gallery_items(saved_paths),
                f"第 {round_index} 轮连接失败：{e}；累计耗时 {format_duration(time.perf_counter() - total_started_at)}。",
            )
            return
        except Exception as e:
            yield (
                "\n\n".join(prompt_history),
                build_gallery_items(saved_paths),
                f"第 {round_index} 轮处理失败：{e}；累计耗时 {format_duration(time.perf_counter() - total_started_at)}。",
            )
            return

    yield (
        "\n\n".join(prompt_history),
        build_gallery_items(saved_paths),
        f"自我迭代完成：共生成 {len(saved_paths)} 张，迭代 {iteration_count} 轮；{format_generation_stats(image_records, iteration_count, time.perf_counter() - total_started_at, request_size)}；品质 {quality}；目录 {save_dir}",
    )


def save_settings(
    save_dir,
    base_url,
    model_id,
    quality,
    api_key,
    random_base_url,
    random_model_id,
    random_api_key,
    random_protocol,
    random_reasoning_effort,
    iteration_base_url,
    iteration_model_id,
    iteration_api_key,
    iteration_protocol,
    iteration_reasoning_effort,
    retry_count,
    retry_delay,
    random_system_prompt,
    random_user_prompt,
    iteration_optimizer_prompt,
    reverse_prompt,
):
    retry_count, retry_delay = normalize_retry_settings(retry_count, retry_delay)
    quality = normalize_quality(quality)
    random_protocol = normalize_protocol(random_protocol)
    iteration_protocol = normalize_protocol(iteration_protocol)
    random_reasoning_effort = normalize_reasoning_effort(random_reasoning_effort)
    iteration_reasoning_effort = normalize_reasoning_effort(iteration_reasoning_effort)

    persist_config(
        {
            "save_dir": save_dir,
            "base_url": base_url,
            "model_id": model_id,
            "quality": quality,
            "api_key": api_key,
            "random_base_url": random_base_url,
            "random_model_id": random_model_id,
            "random_api_key": random_api_key,
            "random_protocol": random_protocol,
            "random_reasoning_effort": random_reasoning_effort,
            "iteration_base_url": iteration_base_url,
            "iteration_model_id": iteration_model_id,
            "iteration_api_key": iteration_api_key,
            "iteration_protocol": iteration_protocol,
            "iteration_reasoning_effort": iteration_reasoning_effort,
            "retry_count": retry_count,
            "retry_delay": retry_delay,
            "random_system_prompt": random_system_prompt,
            "random_user_prompt": random_user_prompt,
            "iteration_optimizer_prompt": iteration_optimizer_prompt,
            "reverse_prompt": reverse_prompt,
        },
    )
    return "设置已保存。"


def load_ui_state():
    latest_config = normalize_config(load_config(DEFAULT_CONFIG))
    CONFIG.clear()
    CONFIG.update(latest_config)
    return [
        latest_config["prompt"],
        latest_config["image_count"],
        latest_config["concurrency"],
        latest_config["aspect_ratio"],
        latest_config["resolution"],
        latest_config["random_preference"],
        latest_config["prompt"],
        latest_config["image_count"],
        latest_config["concurrency"],
        latest_config["aspect_ratio"],
        latest_config["resolution"],
        latest_config["random_preference"],
        latest_config["creative_count"],
        latest_config["text_concurrency"],
        latest_config["image_concurrency"],
        latest_config["aspect_ratio"],
        latest_config["resolution"],
        latest_config["random_preference"],
        latest_config["iteration_prompt_source"],
        latest_config["iteration_custom_prompt"],
        latest_config["prompt"],
        latest_config["iteration_count"],
        latest_config["aspect_ratio"],
        latest_config["resolution"],
        latest_config["save_dir"],
        latest_config["base_url"],
        latest_config["model_id"],
        latest_config["quality"],
        latest_config["api_key"],
        latest_config["random_base_url"],
        latest_config["random_model_id"],
        latest_config["random_protocol"],
        latest_config["random_reasoning_effort"],
        latest_config["random_api_key"],
        latest_config["iteration_base_url"],
        latest_config["iteration_model_id"],
        latest_config["iteration_protocol"],
        latest_config["iteration_reasoning_effort"],
        latest_config["iteration_api_key"],
        latest_config["retry_count"],
        latest_config["retry_delay"],
        latest_config["random_system_prompt"],
        latest_config["random_user_prompt"],
        latest_config["iteration_optimizer_prompt"],
        latest_config["reverse_prompt"],
        "已加载保存的设置。",
    ]


def reset_stop_flag(mode):
    STOP_FLAGS[mode] = False


def request_stop(mode):
    STOP_FLAGS[mode] = True
    return "已请求停止：正在取消排队任务，已开始的网络请求会在当前请求返回后停止继续。"


def should_stop(mode):
    return STOP_FLAGS.get(mode, False)

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="rose",
    neutral_hue="slate",
).set(
    body_background_fill="#f6f2ea",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e6ded2",
    button_primary_background_fill="#243b53",
    button_primary_background_fill_hover="#1b2f43",
    button_primary_text_color="#ffffff",
)

css = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}
.app-shell {
    padding: 18px 8px 28px;
}
.hero {
    min-height: 190px;
    padding: 34px 38px;
    border-radius: 8px;
    background:
        linear-gradient(90deg, rgba(4, 13, 24, .98) 0%, rgba(12, 25, 38, .94) 52%, rgba(81, 45, 34, .82) 100%),
        url("https://images.unsplash.com/photo-1519608487953-e999c86e7455?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    margin-bottom: 16px;
}
.hero h1 {
    color: #ffffff !important;
    font-size: clamp(34px, 4.8vw, 58px);
    line-height: 1.04;
    margin: 0 0 12px;
    letter-spacing: 0;
    font-weight: 850;
    -webkit-text-stroke: .6px rgba(255, 255, 255, .34);
    text-shadow:
        0 2px 0 rgba(0, 0, 0, .34),
        0 8px 24px rgba(0, 0, 0, .72),
        0 0 38px rgba(255, 255, 255, .22);
}
.hero p {
    color: #f8fbff !important;
    max-width: 780px;
    margin: 0;
    color: rgba(255, 255, 255, .92);
    font-size: 17px;
    line-height: 1.7;
    text-shadow: 0 1px 10px rgba(0, 0, 0, .4);
}
.mode-note {
    margin: 0 0 14px;
    padding: 12px 14px;
    border-left: 4px solid #a86240;
    background: #fff8ef;
    color: #4e4035;
    border-radius: 6px;
    line-height: 1.6;
}
.section-title {
    font-size: 13px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #6f6254;
    font-weight: 700;
    margin: 4px 0 10px;
}
.status-box textarea {
    font-weight: 600 !important;
}
.prompt-history-box textarea {
    min-height: 360px !important;
    max-height: 520px !important;
    overflow-y: auto !important;
    resize: vertical !important;
}
.gallery-panel {
    min-height: 0 !important;
}
.gallery-panel .grid-wrap,
.gallery-panel .grid-container,
.gallery-panel .thumbnail-lg {
    min-height: 0 !important;
}
.gallery-panel button[aria-label*="Close"],
.gallery-panel button[title*="Close"],
.gallery-panel button[aria-label*="关闭"],
.gallery-panel button[title*="关闭"] {
    z-index: 10000 !important;
    pointer-events: auto !important;
}
.gallery-panel [role="dialog"],
.gallery-panel .modal,
.gallery-panel .preview {
    z-index: 9999 !important;
}
body:has(:fullscreen) button,
button[aria-label*="Close"],
button[title*="Close"],
button[aria-label*="关闭"],
button[title*="关闭"] {
    pointer-events: auto !important;
}
[role="dialog"],
.modal,
.preview,
.fullscreen {
    z-index: 9999 !important;
}
footer {
    display: none !important;
}
@media (max-width: 720px) {
    .hero {
        padding: 24px 20px;
        min-height: 150px;
    }
}
"""

js = """
() => {
    const isCloseControl = (target) => {
        const control = target.closest("button, [role='button']");
        if (!control) return false;
        const text = (control.innerText || control.textContent || "").trim().toLowerCase();
        const label = (
            control.getAttribute("aria-label") ||
            control.getAttribute("title") ||
            control.getAttribute("data-testid") ||
            ""
        ).toLowerCase();
        return text === "close" || text === "关闭" || label.includes("close") || label.includes("关闭");
    };

    const pressEscape = () => {
        document.dispatchEvent(new KeyboardEvent("keydown", {
            key: "Escape",
            code: "Escape",
            keyCode: 27,
            which: 27,
            bubbles: true,
            cancelable: true,
        }));
    };

    document.addEventListener("click", async (event) => {
        if (!isCloseControl(event.target)) return;

        if (document.fullscreenElement) {
            event.preventDefault();
            event.stopPropagation();
            try {
                await document.exitFullscreen();
            } catch (error) {
                console.debug("Fullscreen exit skipped", error);
            }
            setTimeout(pressEscape, 120);
        }
    }, true);
}
"""

with gr.Blocks(title="GPT Image WebStudio", analytics_enabled=False) as app:
    with gr.Column(elem_classes=["app-shell"]):
        gr.HTML(
            """
            <section class="hero">
                <h1>GPT Image WebStudio</h1>
                <p>面向批量创作的本地工作台：手动提示词、随机抽卡、创意批量、自我迭代与统一接口设置集中管理。</p>
            </section>
            """
        )

        with gr.Tabs():
            with gr.Tab("手动模式"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=340):
                        gr.HTML('<div class="section-title">创作设置</div>')
                        gr.HTML('<div class="mode-note">手动模式：输入一段固定提示词，按设定数量生成图片。适合验证一个明确想法；接口、保存目录和重试参数请在“设置”页统一维护。</div>')
                        prompt_input = gr.Textbox(
                            label="提示词",
                            value=CONFIG["prompt"],
                            placeholder="例如：清晨的玻璃花房里，一位穿白裙的女孩正在照顾蓝色鸢尾花，电影感，细腻光影",
                            lines=7,
                            max_lines=10,
                        )

                        with gr.Row():
                            image_count_input = gr.Slider(
                                label="生成数量",
                                minimum=1,
                                maximum=12,
                                value=CONFIG["image_count"],
                                step=1,
                            )
                            concurrency_input = gr.Slider(
                                label="并发张数",
                                minimum=1,
                                maximum=6,
                                value=CONFIG["concurrency"],
                                step=1,
                            )
                            aspect_ratio_input = gr.Dropdown(
                                label="图片比例",
                                choices=list(ASPECT_RATIOS.keys()),
                                value=CONFIG["aspect_ratio"],
                            )
                            resolution_input = gr.Dropdown(
                                label="分辨率",
                                choices=list(RESOLUTION_PRESETS.keys()),
                                value=CONFIG["resolution"],
                            )

                        with gr.Row():
                            generate_btn = gr.Button("开始生成", variant="primary", size="lg")
                            stop_btn = gr.Button("停止", variant="stop", size="lg")

                    with gr.Column(scale=6, min_width=360):
                        gr.HTML('<div class="section-title">生成结果</div>')
                        gallery_output = gr.Gallery(
                            label="图片画廊",
                            columns=2,
                            rows=1,
                            height="auto",
                            object_fit="contain",
                            show_label=False,
                            allow_preview=True,
                            elem_classes=["gallery-panel"],
                        )
                        status_output = gr.Textbox(
                            label="状态",
                            lines=4,
                            elem_classes=["status-box"],
                        )

            with gr.Tab("随机模式"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=340):
                        gr.HTML('<div class="section-title">随机设置</div>')
                        gr.HTML('<div class="mode-note">随机模式：先由文本模型生成一段随机提示词，再用这段提示词生成多张图片。适合快速抽卡同一主题的多个变化。</div>')
                        random_preference_input = gr.Textbox(
                            label="本次创作方向",
                            value=CONFIG["random_preference"],
                            placeholder="例如：雨后街角、玻璃花房、黄昏海岸",
                            lines=1,
                        )
                        random_prompt_output = gr.Textbox(
                            label="随机提示词",
                            value=CONFIG["prompt"],
                            lines=8,
                            max_lines=12,
                            interactive=False,
                        )

                        with gr.Row():
                            random_image_count_input = gr.Slider(
                                label="生成数量",
                                minimum=1,
                                maximum=12,
                                value=CONFIG["image_count"],
                                step=1,
                            )
                            random_concurrency_input = gr.Slider(
                                label="并发张数",
                                minimum=1,
                                maximum=6,
                                value=CONFIG["concurrency"],
                                step=1,
                            )
                            random_aspect_ratio_input = gr.Dropdown(
                                label="图片比例",
                                choices=list(ASPECT_RATIOS.keys()),
                                value=CONFIG["aspect_ratio"],
                            )
                            random_resolution_input = gr.Dropdown(
                                label="分辨率",
                                choices=list(RESOLUTION_PRESETS.keys()),
                                value=CONFIG["resolution"],
                            )

                        with gr.Row():
                            random_generate_btn = gr.Button("随机生成并出图", variant="primary", size="lg")
                            random_stop_btn = gr.Button("停止", variant="stop", size="lg")

                    with gr.Column(scale=6, min_width=360):
                        gr.HTML('<div class="section-title">生成结果</div>')
                        random_gallery_output = gr.Gallery(
                            label="图片画廊",
                            columns=2,
                            rows=1,
                            height="auto",
                            object_fit="contain",
                            show_label=False,
                            allow_preview=True,
                            elem_classes=["gallery-panel"],
                        )
                        random_status_output = gr.Textbox(
                            label="状态",
                            lines=4,
                            elem_classes=["status-box"],
                        )

            with gr.Tab("创意模式"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=340):
                        gr.HTML('<div class="section-title">创意设置</div>')
                        gr.HTML('<div class="mode-note">创意模式：并发生成多段不同随机提示词，并将每段提示词各生成一张图片。适合夜间批量探索大量创意方向。</div>')
                        creative_preference_input = gr.Textbox(
                            label="本次创作方向",
                            value=CONFIG["random_preference"],
                            placeholder="例如：雨后街角、玻璃花房、黄昏海岸",
                            lines=1,
                        )
                        creative_prompts_output = gr.Textbox(
                            label="随机提示词组",
                            value="",
                            lines=12,
                            max_lines=18,
                            interactive=False,
                        )

                        with gr.Row():
                            creative_count_input = gr.Slider(
                                label="生成张数",
                                minimum=1,
                                maximum=100,
                                value=CONFIG["creative_count"],
                                step=1,
                            )
                            creative_text_concurrency_input = gr.Slider(
                                label="文本并发",
                                minimum=1,
                                maximum=50,
                                value=CONFIG["text_concurrency"],
                                step=1,
                            )
                            creative_image_concurrency_input = gr.Slider(
                                label="图片并发",
                                minimum=1,
                                maximum=12,
                                value=CONFIG["image_concurrency"],
                                step=1,
                            )
                            creative_aspect_ratio_input = gr.Dropdown(
                                label="图片比例",
                                choices=list(ASPECT_RATIOS.keys()),
                                value=CONFIG["aspect_ratio"],
                            )
                            creative_resolution_input = gr.Dropdown(
                                label="分辨率",
                                choices=list(RESOLUTION_PRESETS.keys()),
                                value=CONFIG["resolution"],
                            )

                        with gr.Row():
                            creative_generate_btn = gr.Button("批量创意生成", variant="primary", size="lg")
                            creative_stop_btn = gr.Button("停止", variant="stop", size="lg")

                    with gr.Column(scale=6, min_width=360):
                        gr.HTML('<div class="section-title">生成结果</div>')
                        creative_gallery_output = gr.Gallery(
                            label="图片画廊",
                            columns=2,
                            rows=1,
                            height="auto",
                            object_fit="contain",
                            show_label=False,
                            allow_preview=True,
                            elem_classes=["gallery-panel"],
                        )
                        creative_status_output = gr.Textbox(
                            label="状态",
                            lines=4,
                            elem_classes=["status-box"],
                        )

            with gr.Tab("自我迭代模式"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=340):
                        gr.HTML('<div class="section-title">迭代设置</div>')
                        gr.HTML('<div class="mode-note">自我迭代模式：可随机生成初始提示词，也可手动输入初始提示词。每轮视觉评估都会带上创作主题或初始创作方向，以及本轮图片使用的提示词，尽量避免越迭代越跑题。</div>')
                        iterative_prompt_source_input = gr.Radio(
                            label="初始提示词来源",
                            choices=ITERATION_PROMPT_SOURCE_PRESETS,
                            value=CONFIG["iteration_prompt_source"],
                        )
                        iterative_preference_input = gr.Textbox(
                            label="创作主题" if CONFIG["iteration_prompt_source"] == "自定义提示词" else "初始创作方向",
                            value=CONFIG["random_preference"],
                            placeholder="例如：雨后街角、玻璃花房、黄昏海岸" if CONFIG["iteration_prompt_source"] == "自定义提示词" else "例如：雨后街角、玻璃花房、黄昏海岸",
                            lines=1,
                        )
                        iterative_custom_prompt_input = gr.Textbox(
                            label="初始提示词（点击输入你需要的提示词）" if CONFIG["iteration_prompt_source"] == "自定义提示词" else "初始提示词（由文本模型随机生成）",
                            value=CONFIG["iteration_custom_prompt"] or CONFIG["prompt"],
                            placeholder="像手动模式一样输入第 1 轮要使用的完整提示词",
                            lines=7,
                            max_lines=12,
                            interactive=CONFIG["iteration_prompt_source"] == "自定义提示词",
                        )
                        iterative_prompt_output = gr.Textbox(
                            label="每轮提示词",
                            value=CONFIG["prompt"],
                            lines=12,
                            max_lines=18,
                            interactive=False,
                            elem_classes=["prompt-history-box"],
                        )

                        with gr.Row():
                            iteration_count_input = gr.Slider(
                                label="迭代次数",
                                minimum=1,
                                maximum=6,
                                value=CONFIG["iteration_count"],
                                step=1,
                            )
                            iterative_aspect_ratio_input = gr.Dropdown(
                                label="图片比例",
                                choices=list(ASPECT_RATIOS.keys()),
                                value=CONFIG["aspect_ratio"],
                            )
                            iterative_resolution_input = gr.Dropdown(
                                label="分辨率",
                                choices=list(RESOLUTION_PRESETS.keys()),
                                value=CONFIG["resolution"],
                            )

                        with gr.Row():
                            iterative_generate_btn = gr.Button("开始自我迭代", variant="primary", size="lg")
                            iterative_stop_btn = gr.Button("停止", variant="stop", size="lg")

                    with gr.Column(scale=6, min_width=360):
                        gr.HTML('<div class="section-title">迭代结果</div>')
                        iterative_gallery_output = gr.Gallery(
                            label="图片画廊",
                            columns=2,
                            rows=1,
                            height="auto",
                            object_fit="contain",
                            show_label=False,
                            allow_preview=True,
                            elem_classes=["gallery-panel"],
                        )
                        iterative_status_output = gr.Textbox(
                            label="状态",
                            lines=4,
                            elem_classes=["status-box"],
                        )

            with gr.Tab("提示词反推"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=340):
                        gr.HTML('<div class="section-title">图片输入</div>')
                        gr.HTML('<div class="mode-note">上传图片，或填写本地图片路径。应用会先自动压缩图片，再调用设置页中的多模态模型反推出可复用的中文提示词。</div>')
                        reverse_image_input = gr.Image(
                            label="上传图片",
                            type="filepath",
                            sources=["upload", "clipboard"],
                            height=360,
                        )
                        reverse_local_path_input = gr.Textbox(
                            label="本地图片路径",
                            placeholder=r"例如：D:\Images\sample.png",
                            lines=1,
                        )
                        reverse_generate_btn = gr.Button("开始反推提示词", variant="primary", size="lg")

                    with gr.Column(scale=6, min_width=360):
                        gr.HTML('<div class="section-title">反推结果</div>')
                        reverse_prompt_output = gr.Textbox(
                            label="反推提示词",
                            lines=16,
                            max_lines=24,
                            buttons=["copy"],
                            elem_classes=["prompt-history-box"],
                        )
                        reverse_status_output = gr.Textbox(
                            label="状态",
                            lines=4,
                            elem_classes=["status-box"],
                        )

            with gr.Tab("设置"):
                gr.HTML('<div class="section-title">全局设置</div>')
                gr.HTML('<div class="mode-note">这里集中管理所有模式共用的接口、保存目录、重试策略和提示词模板。修改后点击“保存设置”即可对后续生成生效；如果刷新了页面，请点击“重新读取设置”同步当前配置。也可以关闭正在运行的脚本窗口后重新启动，界面会直接显示已保存的设置。重试策略对文本模型、图片生成、多模态评估和图片 URL 下载全部生效。</div>')

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=360):
                        settings_save_dir_input = gr.Textbox(
                            label="保存目录",
                            value=CONFIG["save_dir"],
                            placeholder="留空则保存到当前项目的 AI_Cards 文件夹",
                        )
                        with gr.Accordion("图片生成接口", open=True):
                            gr.HTML('<div class="mode-note">用途：生成图片。支持 OpenAI Images 兼容接口。示例：https://example.com 或 https://example.com/v1/images/generations</div>')
                            settings_base_url_input = gr.Textbox(label="API 地址", value=CONFIG["base_url"])
                            settings_model_id_input = gr.Textbox(label="模型 ID", value=CONFIG["model_id"])
                            settings_quality_input = gr.Dropdown(
                                label="品质",
                                choices=QUALITY_PRESETS,
                                value=CONFIG["quality"],
                            )
                            settings_api_key_input = gr.Textbox(
                                label="API Key",
                                value=CONFIG["api_key"],
                                type="password",
                            )

                        with gr.Accordion("文本模型接口", open=True):
                            gr.HTML('<div class="mode-note">用途：生成随机提示词。支持 OpenAI Chat、OpenAI Responses、Gemini 原生、Claude Messages；选择“自动识别”时会根据 URL 和模型 ID 判断。建议格式：https://example.com/v1</div>')
                            settings_random_base_url_input = gr.Textbox(label="API 地址", value=CONFIG["random_base_url"])
                            settings_random_model_id_input = gr.Textbox(label="模型 ID", value=CONFIG["random_model_id"])
                            settings_random_protocol_input = gr.Dropdown(
                                label="协议",
                                choices=MODEL_PROTOCOL_PRESETS,
                                value=CONFIG["random_protocol"],
                            )
                            settings_random_reasoning_effort_input = gr.Dropdown(
                                label="思考档位",
                                choices=REASONING_EFFORT_PRESETS,
                                value=CONFIG["random_reasoning_effort"],
                            )
                            settings_random_api_key_input = gr.Textbox(
                                label="API Key",
                                value=CONFIG["random_api_key"],
                                type="password",
                            )

                        with gr.Accordion("多模态模型接口", open=True):
                            gr.HTML('<div class="mode-note">用途：读取已生成图片并优化下一轮提示词。建议使用 Gemini 模型，也支持 OpenAI Chat、OpenAI Responses、Gemini 原生、Claude Messages；选择“自动识别”时会根据 URL 和模型 ID 判断。建议格式：https://example.com</div>')
                            settings_iteration_base_url_input = gr.Textbox(label="API 地址", value=CONFIG["iteration_base_url"])
                            settings_iteration_model_id_input = gr.Textbox(label="模型 ID", value=CONFIG["iteration_model_id"])
                            settings_iteration_protocol_input = gr.Dropdown(
                                label="协议",
                                choices=MODEL_PROTOCOL_PRESETS,
                                value=CONFIG["iteration_protocol"],
                            )
                            settings_iteration_reasoning_effort_input = gr.Dropdown(
                                label="思考档位",
                                choices=REASONING_EFFORT_PRESETS,
                                value=CONFIG["iteration_reasoning_effort"],
                            )
                            settings_iteration_api_key_input = gr.Textbox(
                                label="API Key",
                                value=CONFIG["iteration_api_key"],
                                type="password",
                            )

                        with gr.Accordion("重试设置", open=True):
                            gr.HTML('<div class="mode-note">作用范围：文本模型提示词生成、图片生成、多模态视觉评估、图片 URL 下载。每次请求失败后会等待指定间隔再重试；超过次数后，该任务会报错或跳过，其他可继续的任务不会被中断。</div>')
                            with gr.Row():
                                settings_retry_count_input = gr.Slider(
                                    label="重试次数",
                                    minimum=0,
                                    maximum=5,
                                    value=CONFIG["retry_count"],
                                    step=1,
                                )
                                settings_retry_delay_input = gr.Slider(
                                    label="重试间隔秒",
                                    minimum=0,
                                    maximum=30,
                                    value=CONFIG["retry_delay"],
                                    step=1,
                                )

                    with gr.Column(scale=1, min_width=420):
                        settings_random_system_prompt_input = gr.Textbox(
                            label="文本模型系统提示词（用于提示词生成）",
                            value=CONFIG["random_system_prompt"],
                            lines=8,
                            max_lines=16,
                        )
                        settings_random_user_prompt_input = gr.Textbox(
                            label="文本模型用户提示词（用于提示词生成）",
                            value=CONFIG["random_user_prompt"],
                            lines=8,
                            max_lines=16,
                        )
                        settings_iteration_optimizer_prompt_input = gr.Textbox(
                            label="视觉模型提示词（用于视觉评估迭代模式）",
                            value=CONFIG["iteration_optimizer_prompt"],
                            lines=10,
                            max_lines=20,
                        )
                        settings_reverse_prompt_input = gr.Textbox(
                            label="视觉模型提示词（用于提示词反推）",
                            value=CONFIG["reverse_prompt"],
                            lines=8,
                            max_lines=16,
                        )

                        with gr.Row():
                            settings_save_btn = gr.Button("保存设置", variant="primary", size="lg")
                            settings_reload_btn = gr.Button("重新读取设置", variant="secondary", size="lg")
                        settings_status_output = gr.Textbox(label="保存状态", lines=2)

    manual_event = generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            settings_save_dir_input,
            image_count_input,
            concurrency_input,
            settings_retry_count_input,
            settings_retry_delay_input,
            aspect_ratio_input,
            resolution_input,
            settings_base_url_input,
            settings_model_id_input,
            settings_quality_input,
            settings_api_key_input,
        ],
        outputs=[gallery_output, status_output],
    )
    stop_btn.click(
        fn=lambda: request_stop("manual"),
        outputs=[status_output],
        cancels=[manual_event],
        queue=False,
    )

    random_event = random_generate_btn.click(
        fn=generate_random_image,
        inputs=[
            settings_save_dir_input,
            random_image_count_input,
            random_concurrency_input,
            settings_retry_count_input,
            settings_retry_delay_input,
            random_aspect_ratio_input,
            random_resolution_input,
            settings_base_url_input,
            settings_model_id_input,
            settings_quality_input,
            settings_api_key_input,
            settings_random_base_url_input,
            settings_random_model_id_input,
            settings_random_api_key_input,
            settings_random_protocol_input,
            settings_random_reasoning_effort_input,
            random_preference_input,
        ],
        outputs=[random_prompt_output, random_gallery_output, random_status_output],
    )
    random_stop_btn.click(
        fn=lambda: request_stop("random"),
        outputs=[random_status_output],
        cancels=[random_event],
        queue=False,
    )

    creative_event = creative_generate_btn.click(
        fn=generate_creative_images,
        inputs=[
            settings_save_dir_input,
            creative_count_input,
            creative_text_concurrency_input,
            creative_image_concurrency_input,
            settings_retry_count_input,
            settings_retry_delay_input,
            creative_aspect_ratio_input,
            creative_resolution_input,
            settings_base_url_input,
            settings_model_id_input,
            settings_quality_input,
            settings_api_key_input,
            settings_random_base_url_input,
            settings_random_model_id_input,
            settings_random_api_key_input,
            settings_random_protocol_input,
            settings_random_reasoning_effort_input,
            creative_preference_input,
        ],
        outputs=[creative_prompts_output, creative_gallery_output, creative_status_output],
    )
    creative_stop_btn.click(
        fn=lambda: request_stop("creative"),
        outputs=[creative_status_output],
        cancels=[creative_event],
        queue=False,
    )

    iterative_event = iterative_generate_btn.click(
        fn=generate_iterative_image,
        inputs=[
            settings_save_dir_input,
            iterative_prompt_source_input,
            iterative_custom_prompt_input,
            iteration_count_input,
            settings_retry_count_input,
            settings_retry_delay_input,
            iterative_aspect_ratio_input,
            iterative_resolution_input,
            settings_base_url_input,
            settings_model_id_input,
            settings_quality_input,
            settings_api_key_input,
            settings_random_base_url_input,
            settings_random_model_id_input,
            settings_random_api_key_input,
            settings_random_protocol_input,
            settings_random_reasoning_effort_input,
            iterative_preference_input,
            settings_iteration_base_url_input,
            settings_iteration_model_id_input,
            settings_iteration_api_key_input,
            settings_iteration_protocol_input,
            settings_iteration_reasoning_effort_input,
        ],
        outputs=[iterative_prompt_output, iterative_gallery_output, iterative_status_output],
    )
    iterative_stop_btn.click(
        fn=lambda: request_stop("iterative"),
        outputs=[iterative_status_output],
        cancels=[iterative_event],
        queue=False,
    )

    iterative_prompt_source_input.change(
        fn=update_iteration_source_ui,
        inputs=[iterative_prompt_source_input],
        outputs=[iterative_preference_input, iterative_custom_prompt_input],
        queue=False,
    )

    reverse_event = reverse_generate_btn.click(
        fn=reverse_prompt_from_image,
        inputs=[
            reverse_image_input,
            reverse_local_path_input,
            settings_iteration_base_url_input,
            settings_iteration_model_id_input,
            settings_iteration_api_key_input,
            settings_iteration_protocol_input,
            settings_iteration_reasoning_effort_input,
            settings_retry_count_input,
            settings_retry_delay_input,
        ],
        outputs=[reverse_prompt_output, reverse_status_output],
    )

    ui_state_outputs = [
        prompt_input,
        image_count_input,
        concurrency_input,
        aspect_ratio_input,
        resolution_input,
        random_preference_input,
        random_prompt_output,
        random_image_count_input,
        random_concurrency_input,
        random_aspect_ratio_input,
        random_resolution_input,
        creative_preference_input,
        creative_count_input,
        creative_text_concurrency_input,
        creative_image_concurrency_input,
        creative_aspect_ratio_input,
        creative_resolution_input,
        iterative_preference_input,
        iterative_prompt_source_input,
        iterative_custom_prompt_input,
        iterative_prompt_output,
        iteration_count_input,
        iterative_aspect_ratio_input,
        iterative_resolution_input,
        settings_save_dir_input,
        settings_base_url_input,
        settings_model_id_input,
        settings_quality_input,
        settings_api_key_input,
        settings_random_base_url_input,
        settings_random_model_id_input,
        settings_random_protocol_input,
        settings_random_reasoning_effort_input,
        settings_random_api_key_input,
        settings_iteration_base_url_input,
        settings_iteration_model_id_input,
        settings_iteration_protocol_input,
        settings_iteration_reasoning_effort_input,
        settings_iteration_api_key_input,
        settings_retry_count_input,
        settings_retry_delay_input,
        settings_random_system_prompt_input,
        settings_random_user_prompt_input,
        settings_iteration_optimizer_prompt_input,
        settings_reverse_prompt_input,
        settings_status_output,
    ]

    settings_save_btn.click(
        fn=save_settings,
        inputs=[
            settings_save_dir_input,
            settings_base_url_input,
            settings_model_id_input,
            settings_quality_input,
            settings_api_key_input,
            settings_random_base_url_input,
            settings_random_model_id_input,
            settings_random_api_key_input,
            settings_random_protocol_input,
            settings_random_reasoning_effort_input,
            settings_iteration_base_url_input,
            settings_iteration_model_id_input,
            settings_iteration_api_key_input,
            settings_iteration_protocol_input,
            settings_iteration_reasoning_effort_input,
            settings_retry_count_input,
            settings_retry_delay_input,
            settings_random_system_prompt_input,
            settings_random_user_prompt_input,
            settings_iteration_optimizer_prompt_input,
            settings_reverse_prompt_input,
        ],
        outputs=[settings_status_output],
        queue=False,
    )

    settings_reload_btn.click(
        fn=load_ui_state,
        outputs=ui_state_outputs,
        queue=False,
    )

if __name__ == "__main__":
    app.launch(theme=theme, css=css, js=js, inbrowser=True)
