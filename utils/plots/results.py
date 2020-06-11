import numpy as np
import pandas as pd
from plotnine import *


ggplot(dfmelt, aes(x=factor(round_any(x,0.5)), y=value,fill=variable))+
  geom_boxplot()+
  facet_grid(.~variable)+
  labs(x="X (binned)")+
  theme(axis.text.x=element_text(angle=-90, vjust=0.4,hjust=1))