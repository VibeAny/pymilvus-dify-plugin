import logging
import os
import time
import sys
from typing import Optional

# 从环境变量获取日志级别，默认为INFO
log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
log_level = log_levels.get(log_level_name, logging.INFO)

# 配置日志级别，可通过环境变量LOG_LEVEL控制
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)
logger.info(f"日志级别设置为: {log_level_name}")

from dify_plugin import Plugin, DifyPluginEnv

def create_plugin() -> Plugin:
    """创建插件实例"""
    return Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))

def run_plugin_with_retry(max_retries: int = 5, retry_delay: int = 30) -> None:
    """带重试机制的插件运行"""
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"🚀 Starting Milvus plugin (attempt {retry_count + 1}/{max_retries})")
            
            plugin = create_plugin()
            plugin.run()
            
            # 如果正常退出，不需要重试
            logger.info("✅ Plugin stopped normally")
            break
            
        except KeyboardInterrupt:
            logger.info("⏹️ Plugin stopped by user (Ctrl+C)")
            sys.exit(0)
            
        except Exception as e:
            error_msg = str(e)
            retry_count += 1
            
            # 分析错误类型
            if "handshake failed" in error_msg.lower() or "invalid key" in error_msg.lower():
                logger.warning(f"🔑 Authentication error: {error_msg}")
                if retry_count < max_retries:
                    logger.info(f"⏳ Retrying in {retry_delay} seconds... (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
            elif "connection" in error_msg.lower():
                logger.warning(f"🌐 Connection error: {error_msg}")
                if retry_count < max_retries:
                    logger.info(f"⏳ Retrying in {retry_delay} seconds... (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
            else:
                logger.error(f"❌ Unexpected error: {error_msg}", exc_info=True)
                if retry_count < max_retries:
                    logger.info(f"⏳ Retrying in {retry_delay} seconds... (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
    
    if retry_count >= max_retries:
        logger.error(f"💥 Failed to start plugin after {max_retries} attempts. Exiting.")
        sys.exit(1)

if __name__ == '__main__':
    try:
        run_plugin_with_retry()
    except Exception as e:
        logger.error(f"💥 Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
