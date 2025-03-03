from typing import Callable, Dict, List, Any
import logging
import uuid

class Event:
    """Base event class for the trading system."""
    
    def __init__(self, event_type: str, data: Dict[str, Any] = None):
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = None  # Will be set when dispatched
        self.id = str(uuid.uuid4())

class EventDispatcher:
    """Event dispatcher for the trading system."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
        
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to a specific event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from a specific event type."""
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            
    def dispatch(self, event: Event) -> None:
        """Dispatch an event to all subscribers."""
        if event.event_type not in self.subscribers:
            return
            
        for callback in self.subscribers[event.event_type]:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {str(e)}")