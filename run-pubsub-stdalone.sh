#!/bin/bash

_start=1
_end=20

source /opt/ros/jazzy/setup.bash

server_name='publisher'
client_name='subscriber'

pkill python3
COUNT=0
exp_result="./exp_runners/experiments/message_types-stdalone-1"
mkdir -p $exp_result

python3 exp_runners/standalone/read_runtable.py exp_runners/standalone/run_table.csv > pubsub_remain_runs.txt

run_name=''
msg_type=''
msg_interval=''
msg_size=''

file="pubsub_remain_runs.txt"

if [[ -f "$file" ]]; then
  while IFS= read -r line; do
    read run_name todo type interval size etime cli<<< "$line"
    echo "Client: "$cli
    # Killing remaining processes
    pkill -9 -f python3
    pkill -9 -f $server_name
    pkill -9 -f $client_name
    pkill -9 -f powerjoular
    sleep 1
    echo "Running: msg_type ($type), msg_interval ($interval), msg_size ($size), exec_time($etime)"
    d_folder="${exp_result}/${run_name}"
    echo $d_folder
    rm -Rf $d_folder
    mkdir -p $d_folder

    # Creating measurement files
    echo "exec,label,timestamp,cpu,mem" > $d_folder/server-cpu-mem.csv
    echo "exec,label,timestamp,cpu,mem" > $d_folder/client-cpu-mem.csv
    echo "exec,label,timestamp,cpu" > $d_folder/server-cpu.csv
    echo "exec,label,timestamp,cpu" > $d_folder/client-cpu.csv

    # Custom profiler
    echo "Server..."
    python3 mon-rapl.py -c server -t $etime -f $interval -m $type -s $size -r 0 &> /dev/null & #$d_folder/server-$type-$interval-$size.log &
    #echo "python3 mon-rapl.py -c server -t $etime -f $interval -m $type -s $size -r 1 &> $d_folder/server-$type-$interval-$size.log &"
    SERVER_PID=$!
    echo "Started mon-rapl.py with PID $SERVER_PID"

    echo "Starting $cli clients..."

    for i in $(seq 1 ${cli}); do
      echo "Client..."
      rapl="-r 0"
      # if [ $i -eq $cli ]; then
      #     rapl="-r 1"
      # fi
      python3 mon-rapl.py -c client -m $type $rapl &> /dev/null &
      CLIENT_PID=$!
      echo "Started mon-rapl.py with PID $CLIENT_PID"
    done

    sleep 5

    TALKER_PID=`ps -fC $server_name | tail -1 | grep [0-9] | awk '{ print $2}'`
    ps -fC $server_name
    echo "Server PID: $TALKER_PID"
    LISTENER_PID=`ps -fC $client_name | tail -1 | grep [0-9] | awk '{ print $2}'`
    ps -fC $client_name
    echo "Client PID: $LISTENER_PID"
    
    echo "Running PowerJoular"
    /usr/bin/powerjoular -p $TALKER_PID -f 'energy-server-powerjoular.csv' &
    /usr/bin/powerjoular -p $LISTENER_PID -f 'energy-client-powerjoular.csv' &

    CPU=0.0
    MEM=0
    CPU_L=0.0
    MEM_L=0

    spent_time=0

    while kill -0 $SERVER_PID 2> /dev/null || kill -0 $CLIENT_PID 2> /dev/null || "$(echo "$spent_time < $etime" | bc)" -eq 1; do
        TIME=$(date +%s)
        CURRENT_CPU_PS=`ps -C $server_name -o %cpu | tail -1 | grep [0-9]`
        CURRENT_CPU=`top -bn1 | grep $server_name | tail -1 | awk '{print $9}'`
        CURRENT_CPU=`echo $CURRENT_CPU | sed 's/,/./g'`
        CURRENT_CPU_L_PS=`ps -C $client_name -o %cpu | tail -1 | grep [0-9]`
        CURRENT_CPU_L=`top -bn1 | grep $client_name | tail -1 | awk '{print $9}'` 
        CURRENT_CPU_L=`echo $CURRENT_CPU_L | sed 's/,/./g'`
        CURRENT_MEM=`pmap $SERVER_PID | head -3 | tail -1 | awk '{ print $2 }' | sed 's/K//'`
        CURRENT_MEM_L=`pmap $CLIENT_PID | head -4 | tail -1 | awk '{ print $2 }' | sed 's/K//'`
        CPU=`python3 -c "print (float($CURRENT_CPU))"`
        CPU_L=`python3 -c "print (float($CURRENT_CPU_L))"`
        MEM=`python3 -c "print (float($CURRENT_MEM))"`
        MEM_L=`python3 -c "print (float($CURRENT_MEM_L))"`
        
        echo "$COUNT,talker,$TIME,$CPU,$MEM" >> $d_folder/server-cpu-mem.csv
        echo "$COUNT,listener,$TIME,$CPU_L,$MEM_L" >> $d_folder/client-cpu-mem.csv

        echo "$COUNT,talker,$TIME,$CURRENT_CPU_PS" >> $d_folder/server-cpu.csv
        echo "$COUNT,listener,$TIME,$CURRENT_CPU_L_PS" >> $d_folder/client-cpu.csv
        sleep 0.1
        spent_time=$(echo "$spent_time + 0.5" | bc)
        if [ "$(echo "$spent_time > $etime" | bc)" -eq 1 ]; then
          echo "Timeout!"
          pkill -9 -f $server_name
          pkill -9 -f $client_name
          pkill -9 -f powerjoular
        fi
    done 
    echo "Stopped"
    if [ "$(echo "$spent_time > $etime" | bc)" -eq 1 ]; then
        echo "Timeout!"
    fi
    
    cp -f energy-* $d_folder
    rm -Rf energy-*
    let COUNT++

    # Update the run to DONE
    python3 exp_runners/standalone/update_runtable.py exp_runners/standalone/run_table.csv $run_name

    # Remove ROS logs
    rm -Rf /root/.ros/log/*

    echo "Sleeping..."
    sleep 15
  done < "$file"
else
  echo "File $file not found!"
fi