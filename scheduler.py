
from datetime import datetime, timezone

class SixHourClock:
    @staticmethod
    def last_6h_close(now_utc: datetime) -> datetime:
        hour = (now_utc.hour // 6) * 6
        return now_utc.replace(hour=hour, minute=0, second=0, microsecond=0)

    def new_6h_close(self, now_utc: datetime, last_seen_close) -> bool:
        current = self.last_6h_close(now_utc)
        return last_seen_close is None or current > last_seen_close
