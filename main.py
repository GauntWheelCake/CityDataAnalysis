from heat_map import matplotlib_main, folium_main
from utils import *


if __name__ == '__main__':
    if function == "matplotlib":
        matplotlib_main()
    elif function == "folium":
        folium_main()
    else:
        print("no this option")



