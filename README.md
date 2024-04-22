# Challenges LOG:

## Signal correction:
Even though the dataset provides a filtered signal. some signals seem to be corrupt:

![Alt text](./ECG_ID_dataset/plots/image.png?raw=true "Optional Title")

## Advanced Beat detection:
I want to use individual beats as the input for the model yet some challenges arise.

- Choose the QRS complex vs the P-QRS-T complex?
    - Probably will have to be decided analytically.
- Should I choose a fixed window around the R-peaks and thus provide a constant feature size?
    - Probably choose various window sizes and test analytically.

- What should I do with data imbalance?
    - Some users have 20 records vs 2 records. Should choose 2 records from those with 20 but which?
    - Also, records and beats-per-record is not always the same...
    - [Temporary Solution](#random-record-sampling)


## Random record sampling:

Temporary solution to data imbalance:

```
    def prepare_data(self) -> None:
        for user in self.users_info['users']:
            #Sample 2 random records
            sampled_records = random.sample(range(0, user['num_records']), 2)
            record1 = user['data_paths'][sampled_records[0]]
            record2 = user['data_paths'][sampled_records[1]]

```
```
        for user in self.users_info['users']:
            #Sample min_beats_found random beats
            sampled_beats_indices = random.sample(range(0, len(beats_dict[user['name']])), min_beats_found)
            beats_dict[user['name']]= [beats_dict[user['name']][i] for i in sampled_beats_indices]
```



## Window size = 250. 125 left-rigth around detected peak
![Alt text](./ECG_ID_dataset/plots/image.png?raw=true "Optional Title")