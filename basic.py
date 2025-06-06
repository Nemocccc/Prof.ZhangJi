# -*- coding: utf-8 -*-

from abc import ABC, abstract
import numpy as np

class Aircraft:
    def __init__(self, id, speed):
        self.id = id,
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
    def __init__(self):
        self.WorkMode = None
        self.frequency = None
        self.power = None
        self.erp = None

    @abstractmethod
    def setMode(self, mode):
        """设置工作模式"""
        pass



class EWAircraft(Aircraft):
    def __init__(self, id, speed):
        super.__init__(id, speed)
        self.WorkMode = 'ETS';    # 三种工作模式的表示问题
        self.radar = APY9Radar()  # 使用APY9雷达

    def fly(self, direction):
        self.speed = None   # NOTE
        self.position += direction + self.speed  # NOTE
    
    def radarContrtol(self, mode):
        assert mode in ["ETS", "AW", "ESS"], "无效雷达模式"
        self.radar.set_mode(mode)

    
class APY9Radar(Radar):
    MODE_CONFIG = {
        "ETS": {"scan_interval": 0, "erp": 0, "freq": 0},  
        "AW": {"scan_interval": 0, "erp": 0, "freq": 0},  
        "ESS": {"scan_interval": 0, "erp": 0, "freq": 0}   
    }
    
    def set_mode(self, mode):
        self.current_mode = mode
        config = self.MODE_CONFIG[mode]
        self.scan_interval = config["scan_interval"]  
        self.erp = config["erp"]  
        self.frequency = config["freq"]  


class F35(Aircraft):
    pass 

class F18(Aircraft):
    pass

class FCRadar(Radar):
    """火控雷达实现"""
    MODE_CONFIG = {
        "MTT": {"range": 0, "erp": 0, "freq": 0},  
        "STT": {"range": 0, "erp": 0, "freq": 0}   
    }
    
    def set_mode(self, mode):
        self.current_mode = mode
        config = self.MODE_CONFIG[mode]
        self.detection_range = config["range"]  
        self.erp = config["erp"] 
        self.frequency = config["freq"] 


if __name__ == '__main__':
    pass