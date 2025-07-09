from ultralytics import YOLO, checks, hub

if __name__ == '__main__':
    checks()
    
    hub.login('f74f217c37d7862d7892cf19503c0abeed60d7087e')
    
    model = YOLO('https://hub.ultralytics.com/models/BwvZftBIZwmvZaxVh1dH')
    results = model.train()

