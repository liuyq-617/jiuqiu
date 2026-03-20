"""
简单的内存缓存模块
用于缓存低频变化但高频访问的数据（如负责人列表、客户列表）
"""
from datetime import datetime, timedelta
from typing import Optional, Callable, TypeVar, Dict, Tuple

T = TypeVar('T')


class SimpleCache:
    """简单的 TTL 缓存，线程不安全（适用于单进程场景）"""

    def __init__(self, ttl_seconds: int = 300):
        """
        Args:
            ttl_seconds: 缓存过期时间（秒），默认 5 分钟
        """
        self._cache: Dict[str, Tuple[T, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str, fetch_fn: Callable[[], T]) -> T:
        """
        获取缓存值，如果不存在或已过期则调用 fetch_fn 获取并缓存。

        Args:
            key: 缓存键
            fetch_fn: 获取数据的函数（仅在缓存未命中时调用）

        Returns:
            缓存的值或新获取的值
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._ttl:
                return value

        # 缓存未命中或已过期，重新获取
        value = fetch_fn()
        self._cache[key] = (value, datetime.now())
        return value

    def invalidate(self, key: Optional[str] = None):
        """
        清除缓存。

        Args:
            key: 指定键则清除该键，否则清除所有缓存
        """
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

    def set(self, key: str, value: T):
        """
        直接设置缓存值。

        Args:
            key: 缓存键
            value: 缓存值
        """
        self._cache[key] = (value, datetime.now())


# 全局元数据缓存实例（TTL 5 分钟）
metadata_cache = SimpleCache(ttl_seconds=300)
