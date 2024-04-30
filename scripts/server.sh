#!/bin/bash

## CUSTOM MODIFICATIONS

# Python binding dirs for kmsxx and libcamera
export PYTHONPATH="$PYTHONPATH:/usr/local/lib/aarch64-linux-gnu/python3.11/site-packages"

# Virtualenvwrapper settings
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh

PID="/home/ramin/pyserver.pid"
APP="/opt/server.py"

start() {
    if [ -f $PID ]; then
        echo "Service is running"
    else
        workon "lensflow"
        nohup python3 $APP &> /dev/null &
        echo $! > $PID
        echo "Service started"
    fi
}

stop() {
    if [ -f $PID ]; then
        kill $(cat $PID)
        rm $PID
        echo "Service stopped"
    else
        echo "Service not running"
    fi
}

restart() {
    stop
    start
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
esac