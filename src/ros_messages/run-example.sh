python3 publisher.py --message_type Image --interval 1.0 --message_size 3 --execution_time 10 &
sleep 1
python3 subscriber.py --message_type Image

