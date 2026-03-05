class Settings(BaseSettings):
    """
    系统配置类
    支持从环境变量或 .env 文件读取配置
    """
    
    # ==================== LLM Configuration ====================
    # 主要模型 (Qwen3-32B)
    LLM_BASE_URL: str = "http://localhost:6003/v1"
    LLM_API_KEY: str = "1e174CY6rKs28HcNjxhv"
    LLM_MODEL_NAME: str = "model/Qwen3-32B"
    
    # 轻量模型 (Qwen3-8B)
    LLM_LIGHT_BASE_URL: str = "http://localhost:6002/v1"
    LLM_LIGHT_API_KEY: str = "1e174CY6rKs28HcNjxhv"
    LLM_LIGHT_MODEL_NAME: str = "model/Qwen3-8B"

    # DeepSeek模型 (DeepSeek-V3.1) - 用于全文生成
    DEEPSEEK_BASE_URL: str = "http://10.0.8.11:50019/v1"
    DEEPSEEK_API_KEY: str = "1e174CY6rKs28HcNjxhv"
    DEEPSEEK_MODEL_NAME: str = "/models/Intel_DeepSeek-V3.1-int4-mixed-AutoRound"

    # Distill_Qwen_32B 模型 (DeepSeek-R1-Distill-Qwen-32B)
    DISTILL_QWEN_32B_BASE_URL: str = "http://localhost:6007/v1"
    DISTILL_QWEN_32B_API_KEY: str = "1e174CY6rKs28HcNjxhv"
    DISTILL_QWEN_32B_MODEL_NAME: str = "deepseek-r1-distill-qwen-32b"
    FULL_CONTENT_WRITER_USE_DEEPSEEK: bool = True