# ================= PREPROCESSING =================
def preprocess_data(dataset):
    dataset = dataset.dropna()

    for col in dataset.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y




# ================= HYBRID OPTIMIZATION =================
def fitness_function(solution, dataset):
    numerical_dataset = dataset.select_dtypes(include=np.number).astype(float)
    dataset_mean = numerical_dataset.iloc[:, :len(solution)].mean(axis=0)
    return np.mean((solution - dataset_mean) ** 2)


def initialize_population(pop_size, dim, search_space):
    return np.random.uniform(search_space[0], search_space[1], (pop_size, dim))


def hybrid_optimization(search_space, dataset, num_iterations, pop_size, dim):

    whales = initialize_population(pop_size, dim, search_space)

    for t in range(num_iterations):
        fitness = np.array([fitness_function(w, dataset) for w in whales])
        best = whales[np.argmin(fitness)]

        for i in range(pop_size):
            whales[i] = best + np.random.uniform(-0.1, 0.1, dim)

    return best





# ================= MAIN =================
def main():

    dataset = pd.read_csv("data.csv")  # user will provide

    X, y = preprocess_data(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # reshape for LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Hybrid optimization
    hybrid_optimization([0,1], dataset, 50, 20, 10)

    # Models
    lstm_results = train_lstm(X_train_lstm, y_train, X_test_lstm, y_test)
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test)

    print("\n===== RESULTS =====")
    print("LSTM:", lstm_results)
    print("XGBoost:", xgb_results)


if __name__ == "__main__":
    main()
