import logging
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.power_switch import PowerSwitch

from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

from cflib.positioning.motion_commander import MotionCommander

# URI to the Crazyflie to connect to
uri = 'radio://0/80/2M/E7E7E7E7E7'   #TODO: change the uri based on your crazyflie number
'''
1: radio://0/80/2M/E7E7E7E701
2: radio://0/80/2M/E7E7E7E702
3: radio://0/80/2M/E7E7E7E703
4: radio://0/90/2M/E7E7E7E704
5: radio://0/90/2M/E7E7E7E705
6: radio://0/90/2M/E7E7E7E706
7: radio://0/100/2M/E7E7E7E707
8: radio://0/100/2M/E7E7E7E708
9: radio://0/100/2M/E7E7E7E709
'''

deck_attached_event = Event()

def simple_connect():
    # Test connection. You can use this function to check whether you can connect to crazyflie

    print("Yeah, I'm connected! :D")
    time.sleep(3)
    print("Now I will disconnect :'(")

logging.basicConfig(level=logging.ERROR)

def read_parameter(scf, logconf):
    # TODO: read roll, pitch, yaw information from the crazyflie and print them in the terminal
    
    # Parameters:
    #       scf: SyncCrazyflie object, a synchronous Crazyflie instance
    #       logconf: log configuration


def moving_up(mc, dis):
    # TODO: move the crazyflie up
    
    # Parameters:
    #       mc: motion commander
    #       dis: a floating number representing move up distance


def moving_down(mc, dis):
    # TODO: move the crazyflie down
    
    # Parameters:
    #       mc: motion commander
    #       dis: a floating number representing move down distance


def forwarding(mc, dis):
    # TODO: move the crazyflie forward
    
    # Parameters:
    #       mc: motion commander
    #       dis: a floating number representing forward distance


def backwarding(mc, dis):
    # TODO: move the crazyflie backward
    
    # Parameters:
    #       mc: motion commander
    #       dis: a floating number representing backward distance


def turning_left(mc, deg):
    # TODO: turn the crazyflie left
    
    # Parameters:
    #       mc: motion commander
    #       deg: a floating number representing the degree to turn left


def turning_right(mc, deg):
    # TODO: turn the crazyflie right
    
    # Parameters:
    #       mc: motion commander
    #       deg: a floating number representing the degree to turn right


def landing(mc):
    # land the crazyflie
    
    # Parameters:
    #       mc: motion commander

    mc.stop()


def param_deck_flow(_, value_str):
    # Check whether positioning deck is connected or not

    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')


def fly_commander(scf, lg_stab, mc, SLEEP_TIME = 0.3):
    # Control the crazyflie to following the input command after taking off and before landing
    
    command = ""
    while(command != 'e'):
        command = input()
        command = command.strip()
        if command == 'e':
            break
        elif command == 'i':     # read parameter
            read_parameter(scf, lg_stab)
        elif command[0] == 'u':  # up
            dis = float(command.split()[1])
            moving_up(mc, dis)
        elif command[0] == 'd':  # down
            dis = float(command.split()[1])
            moving_down(mc, dis)
        elif command[0] == 'f':  # forward
            dis = float(command.split()[1])
            forwarding(mc, dis)
        elif command[0] == 'b':  # backward
            dis = float(command.split()[1])
            backwarding(mc, dis)
        elif command[0] == 'l':     # turn left
            deg = float(command.split()[1])
            turning_left(mc, deg)
        elif command[0] == 'r':     # turn right
            deg = float(command.split()[1])
            turning_right(mc, deg)
        elif command == 'n':         # land
            landing(mc)
            return


def base_commander(scf, lg_stab, DEFAULT_HEIGHT = 0.5, SLEEP_TIME = 0.3):
    # Control the crazyflie to following the input command

    command = ""
    mc = None
    print("input command")
    while(command != 'e'):
        command = input()
        command = command.strip()
        if command == 'e':
            break
        elif command == 'i':     # read parameter
            read_parameter(scf, lg_stab)
        elif command == 's':     # take off
            print("crazyflie takes off")
            #TODO: let the crazyflie to take off
            #      You can call fly_commander(scf, lg_stab, mc) to deal with flying part
            
        
if __name__ == '__main__':
    cflib.crtp.init_drivers()

    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')

    group = 'stabilizer'
    name = 'estimator'
    
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        
        scf.cf.param.add_update_callback(group="deck", name="bcLighthouse4",
                                cb=param_deck_flow)
        time.sleep(1)
        
        take_off_simple(scf, lg_stab)
        