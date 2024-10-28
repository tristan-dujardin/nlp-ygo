import argparse
import pickle
import pandas as pd

from src.data_helper import prep_data_predict, prep_data_predict

def get_inputs():
    parser = argparse.ArgumentParser(description="Process names and IDs.")

    parser.add_argument('-n', '--names', nargs='*', 
                        help="List of names separated by spaces (e.g., 'CardA CardB')")
    parser.add_argument('-i', '--ids', nargs='*', type=int, 
                        help="List of IDs separated by spaces (e.g., 142 458)")

    args = parser.parse_args()

    names = args.names if args.names is not None else []
    ids = args.ids if args.ids is not None else []

    return names, ids


def print_results(res_arch, res_ban, test_cards, arch_test):
    print(f"{'Name':<42} {'Expected Banlist':<25} {'Predicted Banlist':<21} {'Expected Archetype':<27} {'Predicted Archetype':<25}")

    for i, card in test_cards.iterrows():
        name = card['name']
        
        expected_ban_str = str(card['ban_tcg'])
        
        arch_index = arch_test.index[arch_test['id'] == card['id']]

        ban_result_str = str(res_ban[i])

        expected_arch_str = str(card['archetype']) if pd.notna(card['archetype']) else 'N/A'
        
        arch_result_str = str(res_arch[arch_index - 1])

        print(f"{name.ljust(42)} expected BL: {expected_ban_str.ljust(25)} pred BL: {ban_result_str.ljust(21)} "
              f"expected arch: {expected_arch_str.ljust(27)} pred arch: {arch_result_str.ljust(25)}")


    
def main():
    input_names, input_ids = get_inputs()

    full_data = pd.read_csv('yu-gi-oh/data/cards.csv')
    test_cards = full_data[full_data['id'].isin(input_ids) | full_data['name'].isin(input_names)].reset_index(drop=False, inplace=False)
    with open('models/archetype_vectorizer.pkl', 'rb') as f:
        archetype_vectorizer = pickle.load(f)
    with open('models/banlist_vectorizer.pkl', 'rb') as f:
        banlist_vectorizer = pickle.load(f)

    arch_test, ban_test = prep_data_predict(test_cards)

    test_vect_archetype = archetype_vectorizer.transform(arch_test['desc'])
    test_vect_banlist = banlist_vectorizer.transform(ban_test['desc'])

    with open('models/archetype_model.pkl', 'rb') as f:
        arch_clf = pickle.load(f)
    with open('models/banlist_model.pkl', 'rb') as f:
        ban_clf = pickle.load(f)


    res_arch = arch_clf.predict(test_vect_archetype)
    res_ban = ban_clf.predict(test_vect_banlist)
    
    print_results(res_arch, res_ban, test_cards, arch_test)

if __name__ == "__main__":
    main()