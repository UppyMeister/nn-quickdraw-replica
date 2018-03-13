from datetime import datetime
from LogLevel import LogLevel

class Logger:
    def __init__(self, logLevel):
        self.logLevel = logLevel

    def Log(self, message, severity = LogLevel.INFO):
        if (severity.value <= self.logLevel.value):
            print("[" + str(datetime.now().time().replace(microsecond=0)) + "][" + severity.name + "] " + str(message))
