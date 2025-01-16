import pandas as pd
import json


df = pd.read_csv('data/dataset/trigger_action_channel_func.csv')

action_list = sorted(df['ACTION_channel'].unique(), key=str.lower)
trigger_list = sorted(df['TRIGGER_channel'].unique(), key=str.lower)

category = ''

action_services_df = pd.DataFrame({
    'service': action_list,
    'category': [category] * len(action_list)
})
trigger_services_df = pd.DataFrame({
    'service': trigger_list,
    'category': [category] * len(trigger_list)
})

action_services_df.to_csv('data/services/action_services.csv', index=False)
trigger_services_df.to_csv('data/services/trigger_services.csv', index=False)

action_function_dict = {}
trigger_function_dict = {}

for idx, row in df.iterrows():
    action = row['ACTION_channel']
    trigger = row['TRIGGER_channel']
    action_function = row['ACTION_function']
    trigger_function = row['TRIGGER_function']
    
    if action not in action_function_dict:
        action_function_dict[action] = []
    if trigger not in trigger_function_dict:
        trigger_function_dict[trigger] = []
        
    if action_function not in action_function_dict[action]:
        action_function_dict[action].append(action_function)
    if trigger_function not in trigger_function_dict[trigger]:
        trigger_function_dict[trigger].append(trigger_function)
    
with open("data/services/action_functions.json", "w") as f:
    json.dump({key: action_function_dict[key] for key in sorted(action_function_dict)}, f, indent=4)
    
with open("data/services/trigger_functions.json", "w") as f:
    json.dump({key: trigger_function_dict[key] for key in sorted(trigger_function_dict)}, f, indent=4)