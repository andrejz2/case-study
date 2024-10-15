import pickle
part_lookup_table = {}
with open("tools/part_lookup.pkl", "wb") as f:
    pickle.dump(part_lookup_table, f)