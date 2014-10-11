#!/bin/sh
#Mirar tema parametres!
python $1 $2


#msg=$1" finished: "

ret=$?


[ $ret -ne 0 ] && notify-send "Program Result" "${msg} ERROR!"
[ $ret -eq 0 ] && notify-send "Program Result" "${msg} OK!"

