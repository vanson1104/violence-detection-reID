from .model import ft_net
from ..config import CONFIG

__all__ = ["ft_net", "CONFIG"]

classifiers = ft_net(CONFIG["nclasses"], stride=CONFIG["stride"] ,linear_num=CONFIG["linear_num"])