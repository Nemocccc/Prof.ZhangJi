from abc import ABC, abstractmethod
import numpy as np
import Threading
import time

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

        self.timer = None
        self.isTracking = False
        self.startTime = None

    def setMode(self, mode):
        self.set_mode(mode)

    def fire(self, isTracking: bool):
        def _timeOutCallback():
            if self.isTracking:
                self.isTracking = False
                print(f"雷达 {self.current_mode} 模式下，目标跟踪成功，导弹发射。")
            self.startTime = None

        def _stopTimer():
            if self.timer:
                self.timer.cancel()
                self.timer = None
            self.isTracking = False
            self.startTime = None

        if isTracking:
            if not self.isTracking:
                self.isTracking = True
                self.startTime = time.time()

                self.timer = threading.Timer(5.0, _timeOutCallback)
                self.timer.start()
                print('计时开始于:', self.startTime)
            else:
                print('跟踪失败。')
        else:
            print('跟踪失败，停止计时。')
            if self.isTracking:
                _stopTimer()
                elapsedTime = time.time() - self.startTime
                print(f"成功跟踪了{elapsed_time:.2f}秒")


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
    def __init__(self):
        pass

    def fly(self, direction):
        pass

    def radarControl(self, mode):
        pass


class F18(Aircraft):
    def __init__(self, id, speed):
        super().__init__(id, speed)
        self.radar = FCRadar()
        self.radar.set_mode("MTT")  # 设置初始工作模式
        self.position = np.zeros(2)  # 初始化位置

    def fly(self, direction):
        pass

    def radarControl(self, mode):
        pass


if __name__ == '__main__':
    pass