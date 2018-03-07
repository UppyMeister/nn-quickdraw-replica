from datetime import datetime

def Log(message, severity = "INFO"):
    print("[" + str(datetime.now().time().replace(microsecond=0)) + "][" + severity + "] " + str(message))
