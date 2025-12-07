from ydata_profiling import ProfileReport
import Data

def profile(filename="train.csv", folder="../Data"):
    loader = Data.Data(filename, folder)
    df = loader.data

    profile = ProfileReport(
        df,
        title="Profile Report",
        explorative=True,
        minimal=True
    )

    profile.to_file("profile.html")

if __name__ == "__main__":
    profile("train.csv", "../Data")