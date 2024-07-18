from ui.app_ui import run_app
import os
from api.network_registration import get_computer_name_and_ip, register_computer, get_computer_id, save_computer_id


def main():
  computer_id = get_computer_id()
  if not computer_id:
    computer_id = register_computer()
    if computer_id:
      save_computer_id(computer_id)
    else:
      print("Falied to register computer.")
      return
    
  run_app()
    
    
if __name__ == "__main__":
  main()