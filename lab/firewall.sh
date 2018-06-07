#!/bin/bash

lab=143

wget http://maups.github.io/lab/$lab.txt -O /tmp/tmp.txt
while [[ $(cat /tmp/tmp.txt) != "YES" && $(cat /tmp/tmp.txt) != "NO" ]]
do
	sleep 1
	wget http://maups.github.io/lab/$lab.txt -O /tmp/tmp.txt
done

if [[ $(cat /tmp/tmp.txt) == "YES" ]]
then
	iptables -A INPUT -s 192.168.141.1 -j ACCEPT
	iptables -A OUTPUT -d 192.168.141.1 -j ACCEPT
	iptables -A INPUT -s 200.128.51.30 -j ACCEPT
	iptables -A OUTPUT -d 200.128.51.30 -j ACCEPT
	iptables -P INPUT DROP
	iptables -P OUTPUT DROP

	rm /tmp/tmp.txt

	while (( 1 ))
	do
		if [[ -f /root/restart.txt && $(cat /root/restart.txt) == "YES" && $(who |grep "^guest") != "" ]]
		then
			shutdown -r now
		fi

		sleep 1
		ls /dev/sd* > /root/check.txt
		if [[ $(diff /root/pattern.txt /root/check.txt) != "" ]]
		then
			echo "YES" > /root/restart.txt
			shutdown -r now
		fi 
	done
else
	rm /root/restart.txt
fi
rm /tmp/tmp.txt

