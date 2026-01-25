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

    # Custom profiler
    echo "Server..."
    python3 mon-rapl.py -c server -t $etime -f $interval -m $type -s $size -r 0 &> mon-rapl-server.log & 
    SERVER_PID=$!
    echo "Started mon-rapl.py with PID $SERVER_PID"

    echo "Starting $cli clients..."

    for i in $(seq 1 ${cli}); do
      echo "Client..."
      rapl="-r 0"
      python3 mon-rapl.py -c client -m $type $rapl &> mon-rapl-client.log &
      CLIENT_PID=$!
      echo "Started mon-rapl.py with PID $CLIENT_PID"
    done

    sleep 2

    # TALKER_PID=`ps -fC $server_name | tail -1 | grep [0-9] | awk '{ print $2}'`
    # ps -fC $server_name
    TALKER_PID=`cat /tmp/publisher.pid`
    echo "Publisher PID: $TALKER_PID"
    LISTENER_PID=`cat /tmp/listener.pid`
    # LISTENER_PID=`ps -fC $client_name | tail -1 | grep [0-9] | awk '{ print $2}'`
    # ps -fC $client_name
    echo "Subscriber PID: $LISTENER_PID"
    
    echo "Running PowerJoular"
    {
      /usr/bin/powerjoular -p "$TALKER_PID"   -f energy-server-powerjoular.csv   &
      srv_mon=$!
      /usr/bin/powerjoular -p "$LISTENER_PID" -f energy-client-powerjoular.csv   &
      cli_mon=$!
      wait "$srv_mon" "$cli_mon"
    } &

    spent_time=0

    while [ "$(echo "$spent_time < $etime" | bc -l)" -eq 1 ]; do
        TIME=$(date +%s)
        sleep 0.1
        spent_time=$(echo "$spent_time + 0.5" | bc)
        if [ "$(echo "$spent_time > $etime" | bc)" -eq 1 ]; then
          echo "Timeout!"
          kill $SERVER_PID
          kill $CLIENT_PID
          pkill -9 -f python3
          pkill -9 -f powerjoular
        fi
    done 
    echo "Stopped"
    if [ "$(echo "$spent_time > $etime" | bc)" -eq 1 ]; then
        echo "Timeout!"
    fi

    pkill -9 -f python3
    pkill -9 -f powerjoular

    sleep 5
    
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