import pandas as pd

train_df_trigger_noisy = pd.read_csv('data/dataset/train_trigger.csv')
train_df_action_noisy = pd.read_csv('data/dataset/train_action.csv')

test_df_trigger_noisy = pd.read_csv('data/dataset/noisy/test_trigger.csv')
test_df_action_noisy  = pd.read_csv('data/dataset/noisy/test_action.csv')

train_df_trigger_noisy['trigger_service'] = train_df_trigger_noisy['output'].str.extract(r'TRIGGER SERVICE:\s*([^,]+)')
train_df_trigger_noisy['trigger_event'] = train_df_trigger_noisy['output'].str.extract(r'TRIGGER EVENT:\s*(.+)')
test_df_trigger_noisy['trigger_service'] = test_df_trigger_noisy['output'].str.extract(r'TRIGGER SERVICE:\s*([^,]+)')
test_df_trigger_noisy['trigger_event'] = test_df_trigger_noisy['output'].str.extract(r'TRIGGER EVENT:\s*(.+)')

train_df_action_noisy['action_service'] = train_df_action_noisy['output'].str.extract(r'ACTION SERVICE:\s*([^,]+)')
train_df_action_noisy['action_event'] = train_df_action_noisy['output'].str.extract(r'ACTION EVENT:\s*(.+)')
test_df_action_noisy['action_service'] = test_df_action_noisy['output'].str.extract(r'ACTION SERVICE:\s*([^,]+)')
test_df_action_noisy['action_event'] = test_df_action_noisy['output'].str.extract(r'ACTION EVENT:\s*(.+)')

# Count and print unique elements for train_df
print(f"Train DataFrame: {len(train_df_trigger_noisy)}")
print(f"Unique trigger_service: {train_df_trigger_noisy['trigger_service'].nunique()}")
print(f"Unique trigger_event: {train_df_trigger_noisy['trigger_event'].nunique()}")
print(f"Unique action_service: {train_df_action_noisy['action_service'].nunique()}")
print(f"Unique action_event: {train_df_action_noisy['action_event'].nunique()}")

# Count and print unique elements for test_df
print(f"\nTest DataFrame Noisy: {len(test_df_trigger_noisy)}")
print(f"Unique trigger_service: {test_df_trigger_noisy['trigger_service'].nunique()}")
print(f"Unique trigger_event: {test_df_trigger_noisy['trigger_event'].nunique()}")
print(f"Unique action_service: {test_df_action_noisy['action_service'].nunique()}")
print(f"Unique action_event: {test_df_action_noisy['action_event'].nunique()}")


test_df_trigger_gold = pd.read_csv('data/dataset/gold/test_trigger_one_shot.csv')
test_df_action_gold  = pd.read_csv('data/dataset/gold/test_action_one_shot.csv')

test_df_trigger_gold['trigger_service'] = test_df_trigger_gold['output'].str.extract(r'TRIGGER SERVICE:\s*([^,]+)')
test_df_trigger_gold['trigger_event'] = test_df_trigger_gold['output'].str.extract(r'TRIGGER EVENT:\s*(.+)')

test_df_action_gold['action_service'] = test_df_action_gold['output'].str.extract(r'ACTION SERVICE:\s*([^,]+)')
test_df_action_gold['action_event'] = test_df_action_gold['output'].str.extract(r'ACTION EVENT:\s*(.+)')

# Count and print unique elements for test_df
print(f"\nTest DataFrame Gold: {len(test_df_trigger_gold)}")
print(f"Unique trigger_service: {test_df_trigger_gold['trigger_service'].nunique()}")
print(f"Unique trigger_event: {test_df_trigger_gold['trigger_event'].nunique()}")
print(f"Unique action_service: {test_df_action_gold['action_service'].nunique()}")
print(f"Unique action_event: {test_df_action_gold['action_event'].nunique()}")

#sample noisy train and test and plot the same
val_trigger = train_df_trigger_noisy.sample(1000)
val_action = train_df_action_noisy.sample(1000)

# Count and print unique elements for test_df
print(f"\nVal DataFrame: {len(val_trigger)}")
print(f"Unique trigger_service: {val_trigger['trigger_service'].nunique()}")
print(f"Unique trigger_event: {val_trigger['trigger_event'].nunique()}")
print(f"Unique action_service: {val_action['action_service'].nunique()}")
print(f"Unique action_event: {val_action['action_event'].nunique()}")