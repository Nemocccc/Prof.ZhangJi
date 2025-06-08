from abc import ABC, abstractmethod
import numpy as np

class Aircraft:
    """飞船类"""
    def __init__(self, id, speed):
        self.id = id
        self.speed = speed
        self.radar = None

    @abstractmethod
    def fly(self, direction):
        """怎么飞？"""
        pass

    @abstractmethod
    def radarControl(self, mode):
        """雷达控制"""
        pass


class Radar(ABC):
    """雷达类"""
    def __init__(self):
        self.WorkMode = None
        self.frequency = None
        self.power = None
        self.erp = None

    @abstractmethod
    def setMode(self, mode):
        """设置工作模式"""
        pass


class APY9Radar(Radar):
    """定义不同工作模式下的各种参数：scan_interval扫描周期、erp信号功率、freq工作频率"""
    MODE_CONFIG = {
        "ETS": {"scan_interval": 0, "erp": 100, "freq": 1000},  # 示例值
        "AW": {"scan_interval": 0, "erp": 200, "freq": 2000},
        "ESS": {"scan_interval": 0, "erp": 300, "freq": 3000}
    }

    def set_mode(self, mode):
        """设置工作模式"""
        self.current_mode = mode
        config = self.MODE_CONFIG[mode]
        self.scan_interval = config["scan_interval"]
        self.erp = config["erp"]
        self.frequency = config["freq"]

    def setMode(self, mode):
        self.set_mode(mode)


class FCRadar(Radar):
    """火控雷达实现"""
    MODE_CONFIG = {
        "MTT": {"range": 0, "erp": 100, "freq": 1000},  # 示例值
        "STT": {"range": 0, "erp": 200, "freq": 2000}
    }
    def set_mode(self, mode):
        self.current_mode = mode
        config = self.MODE_CONFIG[mode]
        self.detection_range = config["range"]
        self.erp = config["erp"]
        self.frequency = config["freq"]

    def setMode(self, mode):
        self.set_mode(mode)


class EWAircraft(Aircraft):
    """预警机类"""
    def __init__(self, id, speed):
        super().__init__(id, speed)
        self.WorkMode = 'ETS'
        self.radar = APY9Radar()  # 使用APY9雷达
        self.radar.set_mode(self.WorkMode)  # 设置初始工作模式

    def fly(self, direction):
        self.speed = 0
        self.position += direction + self.speed  # 这里逻辑可能也需要调整

    def radarControl(self, mode):
        assert mode in ["ETS", "AW", "ESS"], "无效雷达模式"
        self.radar.set_mode(mode)


class F35(Aircraft):
    pass


class F18(Aircraft):
    def __init__(self, id, speed):
        super().__init__(id, speed)
        self.radar = FCRadar()
        self.radar.set_mode("MTT")  # 设置初始工作模式
        self.position = np.zeros(2)  # 初始化位置


if __name__ == '__main__':
    pass