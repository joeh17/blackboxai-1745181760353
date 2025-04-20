import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import argparse
import dask.dataframe as dd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import tempfile
import webbrowser

def show_splash_screen():
    splash = tk.Tk()
    splash.overrideredirect(True)
    splash.geometry("400x200+500+300")  # Width x Height + X + Y
    splash.configure(bg='white')

    label = tk.Label(splash, text="Welcome to datalexis", font=("Helvetica", 24), bg='white')
    label.pack(expand=True)

    # Close splash screen after 3 seconds
    splash.after(3000, splash.destroy)
    splash.mainloop()

def show_intro_window(image_path=None, video_path=None):
    intro = tk.Tk()
    intro.title("Intro")

    if image_path:
        img = Image.open(image_path)
        img = img.resize((600, 400), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(intro, image=photo)
        label.image = photo
        label.pack()

    if video_path:
        cap = cv2.VideoCapture(video_path)

        canvas = tk.Canvas(intro, width=600, height=400)
        canvas.pack()

        def play_video():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((600, 400), Image.ANTIALIAS)
                imgtk = ImageTk.PhotoImage(image=img)
                canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                canvas.imgtk = imgtk
                intro.after(30, play_video)
            else:
                cap.release()
                intro.destroy()

        play_video()
        intro.mainloop()
    else:
        # If no video, just show image for 3 seconds
        intro.after(3000, intro.destroy)
        intro.mainloop()

def process_in_chunks(file_path, chunk_size):
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
    summary_stats = None
    head_data = None
    for i, chunk in enumerate(chunk_iter):
        if i == 0:
            head_data = chunk.head()
            summary_stats = chunk.describe()
        else:
            summary_stats = summary_stats.add(chunk.describe(), fill_value=0)
    # Average the summary stats by number of chunks
    summary_stats = summary_stats / (i + 1)
    return head_data, summary_stats

class DataAnalysisApp(tk.Tk):
    def __init__(self, data, dask_mode=False, dataset_path=None):
        super().__init__()
        self.title("Datalexis Interactive Analysis")
        self.geometry("1000x750")
        self.data = data
        self.dask_mode = dask_mode
        self.dataset_path = dataset_path

        self.create_widgets()

    def create_widgets(self):
        tab_control = ttk.Notebook(self)
        self.tab_overview = ttk.Frame(tab_control)
        self.tab_visualization = ttk.Frame(tab_control)
        self.tab_ml = ttk.Frame(tab_control)
        self.tab_data_editor = ttk.Frame(tab_control)
        self.tab_report = ttk.Frame(tab_control)

        tab_control.add(self.tab_overview, text='Overview')
        tab_control.add(self.tab_visualization, text='Visualization')
        tab_control.add(self.tab_ml, text='Machine Learning')
        tab_control.add(self.tab_data_editor, text='Data Editor')
        tab_control.add(self.tab_report, text='Report Generation')
        tab_control.pack(expand=1, fill='both')

        self.create_overview_tab()
        self.create_visualization_tab()
        self.create_ml_tab()
        self.create_data_editor_tab()
        self.create_report_tab()

    def create_overview_tab(self):
        text = tk.Text(self.tab_overview, wrap='word')
        text.pack(expand=1, fill='both')
        overview = "Data Overview:\n\n"
        if self.dask_mode:
            overview += str(self.data.head().compute()) + "\n\n"
            overview += str(self.data.describe().compute())
        else:
            overview += str(self.data.head()) + "\n\n"
            overview += str(self.data.describe())
        text.insert('1.0', overview)
        text.config(state='disabled')

    def create_visualization_tab(self):
        btn_scatter = ttk.Button(self.tab_visualization, text="Show Scatter Plot", command=self.show_scatter_plot)
        btn_scatter.pack(pady=10)

    def show_scatter_plot(self):
        if self.dask_mode:
            df = self.data.sample(frac=0.1).compute()
        else:
            df = self.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            messagebox.showerror("Error", "Not enough numeric columns for scatter plot.")
            return
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot")
        fig.show()

    def create_ml_tab(self):
        lbl = ttk.Label(self.tab_ml, text="Basic Machine Learning Models")
        lbl.pack(pady=10)

        btn_kmeans = ttk.Button(self.tab_ml, text="Run KMeans Clustering", command=self.run_kmeans)
        btn_kmeans.pack(pady=5)

        btn_regression = ttk.Button(self.tab_ml, text="Run Linear Regression", command=self.run_regression)
        btn_regression.pack(pady=5)

        self.ml_output = tk.Text(self.tab_ml, height=15, wrap='word')
        self.ml_output.pack(expand=1, fill='both', pady=10)

    def run_kmeans(self):
        if self.dask_mode:
            df = self.data.sample(frac=0.1).compute()
        else:
            df = self.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            messagebox.showerror("Error", "No numeric columns available for clustering.")
            return
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(df[numeric_cols])
        df['Cluster'] = clusters
        output = f"KMeans clustering done. Cluster counts:\n{df['Cluster'].value_counts()}\n"
        self.ml_output.delete('1.0', tk.END)
        self.ml_output.insert(tk.END, output)

    def run_regression(self):
        if self.dask_mode:
            df = self.data.sample(frac=0.1).compute()
        else:
            df = self.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            messagebox.showerror("Error", "Not enough numeric columns for regression.")
            return
        X = df[[numeric_cols[0]]]
        y = df[numeric_cols[1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        output = f"Linear Regression done.\nR^2 score: {score:.4f}\n"
        self.ml_output.delete('1.0', tk.END)
        self.ml_output.insert(tk.END, output)

    def create_data_editor_tab(self):
        self.tree = ttk.Treeview(self.tab_data_editor)
        self.tree.pack(expand=1, fill='both')

        # Setup columns
        cols = list(self.data.columns)
        self.tree['columns'] = cols
        self.tree['show'] = 'headings'
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        # Insert data
        if self.dask_mode:
            df = self.data.head().compute()
        else:
            df = self.data
        for _, row in df.iterrows():
            self.tree.insert('', 'end', values=list(row))

        btn_save = ttk.Button(self.tab_data_editor, text="Save Changes", command=self.save_changes)
        btn_save.pack(pady=10)

    def save_changes(self):
        cols = list(self.data.columns)
        new_data = []
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            new_data.append(values)
        df_new = pd.DataFrame(new_data, columns=cols)
        save_path = self.dataset_path if self.dataset_path else 'data/edited_data.csv'
        try:
            df_new.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Data saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {e}")

    def create_report_tab(self):
        lbl = ttk.Label(self.tab_report, text="Generate Analysis Report")
        lbl.pack(pady=10)

        btn_generate = ttk.Button(self.tab_report, text="Generate Report", command=self.generate_report)
        btn_generate.pack(pady=10)

    def generate_report(self):
        try:
            temp_dir = tempfile.mkdtemp()
            report_path = os.path.join(temp_dir, "datalexis_report.html")

            # Generate summary
            if self.dask_mode:
                summary = self.data.describe().compute().to_html()
            else:
                summary = self.data.describe().to_html()

            # Generate scatter plot image
            if self.dask_mode:
                df_sample = self.data.sample(frac=0.1).compute()
            else:
                df_sample = self.data
            numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig = px.scatter(df_sample, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot")
                img_path = os.path.join(temp_dir, "scatter_plot.png")
                fig.write_image(img_path)
            else:
                img_path = None

            # Create HTML report
            html_content = f"""
            <html>
            <head><title>Datalexis Analysis Report</title></head>
            <body>
            <h1>Datalexis Analysis Report</h1>
            <h2>Summary Statistics</h2>
            {summary}
            """
            if img_path:
                html_content += f'<h2>Scatter Plot</h2><img src="{img_path}" alt="Scatter Plot"/>'
            html_content += "</body></html>"

            with open(report_path, 'w') as f:
                f.write(html_content)

            webbrowser.open(f'file://{report_path}')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Datalexis Data Analysis")
    parser.add_argument('--dataset', type=str, default='data/sample.csv', help='Path to the dataset CSV file')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting of data')
    parser.add_argument('--chunked', action='store_true', help='Enable chunked reading for large files')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size for chunked reading')
    parser.add_argument('--intro-image', type=str, help='Path to intro image file')
    parser.add_argument('--intro-video', type=str, help='Path to intro video file')
    parser.add_argument('--dask', action='store_true', help='Enable Dask for scalable data processing')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive GUI analysis')
    args = parser.parse_args()

    show_splash_screen()

    if args.intro_image or args.intro_video:
        show_intro_window(args.intro_image, args.intro_video)

    if args.interactive:
        if args.dask:
            data = dd.read_csv(args.dataset)
        elif args.chunked:
            # For interactive mode, chunked reading is not supported; load full data
            data = pd.read_csv(args.dataset)
        else:
            data = pd.read_csv(args.dataset)
        app = DataAnalysisApp(data, dask_mode=args.dask, dataset_path=args.dataset)
        app.mainloop()
        return

    if args.dask:
        try:
            ddf = dd.read_csv(args.dataset)
            print(f"Data loaded with Dask from {args.dataset}.")
            print("First few rows:")
            print(ddf.head())
            print("Summary statistics:")
            print(ddf.describe().compute())
            if not args.no_plot:
                # Sample data for plotting
                sample_df = ddf.sample(frac=0.1).compute()
                sns.pairplot(sample_df)
                plt.show()
        except FileNotFoundError:
            print(f"Dataset not found at {args.dataset}. Please add your dataset to the specified path.")
            return
    elif args.chunked:
        try:
            head_data, summary_stats = process_in_chunks(args.dataset, args.chunk_size)
            print(f"Data loaded in chunks from {args.dataset}.")
            print("First few rows:")
            print(head_data)
            print("Summary statistics (averaged over chunks):")
            print(summary_stats)
        except FileNotFoundError:
            print(f"Dataset not found at {args.dataset}. Please add your dataset to the specified path.")
            return
    else:
        try:
            data = pd.read_csv(args.dataset)
            print(f"Data loaded successfully from {args.dataset}.")
        except FileNotFoundError:
            print(f"Dataset not found at {args.dataset}. Please add your dataset to the specified path.")
            return

        print(data.head())
        print(data.describe())

        if not args.no_plot:
            sns.pairplot(data)
            plt.show()

if __name__ == "__main__":
    main()
