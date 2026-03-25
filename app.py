import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class SalesForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Forecast Prediction App")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load model and data
        try:
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            self.data = pd.read_csv('data.csv')
            self.data['date'] = pd.to_datetime(self.data['date'])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model or data: {e}")
            return
        
        # Set up UI
        self.setup_ui()
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X)
        
        header_label = tk.Label(header_frame, text="Sales Forecast Prediction System", 
                               font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        header_label.pack(pady=15)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(content_frame, bg='white', relief=tk.RIDGE, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        control_label = tk.Label(left_panel, text="Forecast Controls", font=('Arial', 12, 'bold'), bg='white')
        control_label.pack(pady=10)
        
        # Input for number of months
        tk.Label(left_panel, text="Months to Predict:", font=('Arial', 10), bg='white').pack(pady=5)
        self.months_var = tk.IntVar(value=6)
        months_spin = tk.Spinbox(left_panel, from_=1, to=24, textvariable=self.months_var, width=10, font=('Arial', 10))
        months_spin.pack(pady=5)
        
        # Predict button
        predict_btn = tk.Button(left_panel, text="Generate Forecast", command=self.predict_sales,
                               bg='#3498db', fg='white', font=('Arial', 10, 'bold'), padx=20, pady=10)
        predict_btn.pack(pady=15)
        
        # Display current stats
        stats_label = tk.Label(left_panel, text="Data Statistics", font=('Arial', 11, 'bold'), bg='white')
        stats_label.pack(pady=15)
        
        stats_text = f"""
Total Records: {len(self.data)}
Date Range: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}

Latest Sales:
Date: {self.data['date'].iloc[-1].strftime('%Y-%m-%d')}
Sales: {self.data['sales'].iloc[-1]:.2f}

Average Sales: {self.data['sales'].mean():.2f}
Min Sales: {self.data['sales'].min():.2f}
Max Sales: {self.data['sales'].max():.2f}
        """
        stats_display = tk.Label(left_panel, text=stats_text, font=('Arial', 9), 
                                bg='white', justify=tk.LEFT, padx=10)
        stats_display.pack(pady=10, fill=tk.X)
        
        # Right panel - Results and Chart
        right_panel = tk.Frame(content_frame, bg='white', relief=tk.RIDGE, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        result_label = tk.Label(right_panel, text="Forecast Results", font=('Arial', 12, 'bold'), bg='white')
        result_label.pack(pady=10)
        
        # Results table
        self.result_text = tk.Text(right_panel, width=50, height=10, font=('Courier', 9))
        self.result_text.pack(padx=10, pady=5, fill=tk.BOTH)
        
        # Canvas for chart
        self.canvas_frame = tk.Frame(right_panel, bg='white')
        self.canvas_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
    
    def predict_sales(self):
        months_to_predict = self.months_var.get()
        
        # Calculate month numbers for prediction
        last_month = (self.data['date'].dt.year.max() - 2020) * 12 + self.data['date'].dt.month.max()
        future_months = list(range(last_month + 1, last_month + months_to_predict + 1))
        
        # Make predictions
        import numpy as np
        future_months_df = pd.DataFrame({'month': future_months})
        predictions = self.model.predict(future_months_df)
        
        # Display results
        self.result_text.delete(1.0, tk.END)
        result_output = "DATE\t\t\tPREDICTED SALES\n"
        result_output += "=" * 40 + "\n"
        
        for i, pred in enumerate(predictions):
            month_num = future_months[i]
            year = 2020 + (month_num - 1) // 12
            month = ((month_num - 1) % 12) + 1
            date_str = f"{year}-{month:02d}-01"
            result_output += f"{date_str}\t\t{pred:.2f}\n"
        
        self.result_text.insert(tk.END, result_output)
        
        # Plot chart
        self.plot_forecast(future_months, predictions)
    
    def plot_forecast(self, future_months, predictions):
        # Clear previous plot
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot historical data
        ax.plot(self.data.index, self.data['sales'], 'b-', linewidth=2, label='Historical Sales')
        ax.scatter(self.data.index, self.data['sales'], color='blue', s=30, alpha=0.6)
        
        # Plot predictions
        last_index = len(self.data) - 1
        prediction_indices = list(range(last_index + 1, last_index + len(predictions) + 1))
        ax.plot(prediction_indices, predictions, 'r--', linewidth=2, label='Forecasted Sales')
        ax.scatter(prediction_indices, predictions, color='red', s=30, alpha=0.6)
        
        # Add vertical line to separate historical and forecast
        ax.axvline(x=last_index + 0.5, color='gray', linestyle=':', linewidth=1)
        
        # Customize plot
        ax.set_xlabel('Time Period', fontsize=10)
        ax.set_ylabel('Sales', fontsize=10)
        ax.set_title('Sales Forecast Visualization', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        # Display chart
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = SalesForecastApp(root)
    root.mainloop()
