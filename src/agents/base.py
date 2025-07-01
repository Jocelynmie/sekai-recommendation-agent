from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import logging
import time
from datetime import datetime
import json
import hashlib

@dataclass
class AgentResponse:
    """Response from an agent with metadata and token tracking"""
    content: Any
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary for serialization"""
        return {
            'content': self.content,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'tokens_used': self.tokens_used,
            'response_time': self.response_time,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentResponse':
        """Create response from dictionary"""
        data = data.copy()
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class BaseAgent:
    """Base class for all agents in the recommendation system"""
    
    def __init__(self, 
                 agent_id: str,
                 name: str,
                 description: str = "",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name
            description: Agent description
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.config = config or {}
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Token tracking
        self.total_tokens_used = 0
        self.token_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        self.logger.info(f"Initialized {self.name} (ID: {self.agent_id})")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup agent-specific logger"""
        logger = logging.getLogger(f"agent.{self.agent_id}")
        logger.setLevel(logging.INFO)
        
        # Create handler if it doesn't exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _track_tokens(self, tokens_used: int, operation: str = "unknown"):
        """Track token usage"""
        self.total_tokens_used += tokens_used
        token_record = {
            'timestamp': datetime.now(),
            'tokens_used': tokens_used,
            'operation': operation,
            'total_tokens': self.total_tokens_used
        }
        self.token_history.append(token_record)
        self.logger.debug(f"Used {tokens_used} tokens for {operation}")
    
    def _track_request(self, success: bool, response_time: float):
        """Track request performance"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.logger.debug(f"Request completed: success={success}, time={response_time:.3f}s")
    
    def process(self, input_data: Any, **kwargs) -> AgentResponse:
        """
        Process input data and return response
        
        Args:
            input_data: Input data to process
            **kwargs: Additional arguments
            
        Returns:
            AgentResponse with results and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing request with {type(input_data).__name__}")
            
            # Override this method in subclasses
            result = self._process_impl(input_data, **kwargs)
            
            response_time = time.time() - start_time
            self._track_request(True, response_time)
            
            return AgentResponse(
                content=result,
                success=True,
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self._track_request(False, response_time)
            
            self.logger.error(f"Error processing request: {str(e)}", exc_info=True)
            
            return AgentResponse(
                content=None,
                success=False,
                error_message=str(e),
                metadata={'agent_id': self.agent_id, 'agent_name': self.name},
                response_time=response_time
            )
    
    def _process_impl(self, input_data: Any, **kwargs) -> Any:
        """
        Implementation of processing logic - override in subclasses
        
        Args:
            input_data: Input data to process
            **kwargs: Additional arguments
            
        Returns:
            Processed result
        """
        raise NotImplementedError("Subclasses must implement _process_impl")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1),
            'total_tokens_used': self.total_tokens_used,
            'token_history_count': len(self.token_history)
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.total_tokens_used = 0
        self.token_history.clear()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.logger.info("Statistics reset")
    
    def save_state(self, filepath: str):
        """Save agent state to file"""
        state = {
            'agent_id': self.agent_id,
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'stats': self.get_stats(),
            'token_history': [
                {**record, 'timestamp': record['timestamp'].isoformat()}
                for record in self.token_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load agent state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore token history with proper datetime objects
        self.token_history = []
        for record in state.get('token_history', []):
            record['timestamp'] = datetime.fromisoformat(record['timestamp'])
            self.token_history.append(record)
        
        self.logger.info(f"State loaded from {filepath}")
    
    def __str__(self) -> str:
        return f"{self.name} (ID: {self.agent_id})"
    
    def __repr__(self) -> str:
        return f"BaseAgent(agent_id='{self.agent_id}', name='{self.name}')"
