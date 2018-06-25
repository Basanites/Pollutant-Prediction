class Observable:
    def __init__(self, events):
        self.observers = {event: dict()
                          for event in events}

    def get_observers(self, event):
        return self.observers[event]

    def register(self, observer, event, callback):
        self.get_observers(event)[observer] = callback

    def unregister(self, observer, event):
        del self.get_observers(event)[observer]

    def notify(self, event, message):
        for observer, callback in self.get_observers(event).items():
            callback(message)


class Observer:
    def __init__(self):
        pass

    def update(self, message):
        pass
