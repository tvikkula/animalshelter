import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)


def submission(test, fit):
    y_pred = fit.predict_proba(test)

    results = pd.DataFrame(
        {
            'ID': range(1, len(y_pred) + 1),
            'Adoption': y_pred[:,0],
            'Died': y_pred[:,1],
            'Euthanasia': y_pred[:,2],
            'Return_to_owner': y_pred[:,3],
            'Transfer': y_pred[:,4]
        }
    )

    results.to_csv("submission.csv", index=False)
