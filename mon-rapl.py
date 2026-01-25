import os
import argparse
import pyRAPL
import csv

pyRAPL.setup()

parser = argparse.ArgumentParser(description='Energy Monitor - RAPL')
parser.add_argument("-c", "--command", help="Server or client", nargs='?')
parser.add_argument("-f", "--frequency", help="Msg Interval", nargs='?')
parser.add_argument("-t", "--timeout", help="Timeout", nargs='?')
parser.add_argument("-m", "--type", help="Message type", nargs='?')
parser.add_argument("-s", "--size", help="Message size", nargs='?')
parser.add_argument("-r", "--rapl", help="Enable RAPL", nargs='?')
args = parser.parse_args()

def execute_python_file():
   try:
      if args.command == 'server':
         gen_rate = 1 / float(args.frequency)
         command='python3 src/ros_messages/mp_pubsub.py --execution_time '+args.timeout+' --gen_rate '+str(gen_rate)+' --pub_timer '+args.frequency+' --message_type '+args.type+' --message_size '+args.size
         print(f'Command: {command}')
      else:
         # client
        command='python3 src/ros_messages/subscriber.py --message_type '+args.type
      if args.rapl == '1':
         print("RAPL is set")
         meter = pyRAPL.Measurement(args.command[0])   
         meter.begin()
      else:
         print("RAPL is not set")
      print(command)
      os.system(f'{command}')
      if 'meter' in locals():
         meter.end()
         #print(meter.result)
         duration = int(meter.result.duration) / 1000000
         energy_microjoules = float(meter.result.pkg[0])
         energy_joules = energy_microjoules / 1e6
         average_power = float(energy_joules) / duration
         csv_filename = f'energy-{args.command[0]}.csv'
         with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Execution Time','Energy (microjoules)','Energy (joules)', 'Average Power (Watts)'])
            writer.writerow([duration, energy_microjoules, energy_joules, average_power])

   except FileNotFoundError:
      print(f"Error: The command '{command}' is not valid.")

def main():
    execute_python_file()

if __name__ == '__main__':
    main()