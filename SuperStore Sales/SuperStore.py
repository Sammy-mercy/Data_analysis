# Import Necessary Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objects as go

#Load in the data
data = pd.read_excel('SuperStoreUS-2015.xlsx')

#Read data
data

# Get Information about the data
sns.heatmap(data.isnull())

# Fill in the values for the missing data
sns.heatmap(data.drop(['Order Priority', 'Customer Name', 'Ship Mode', 
                       'Customer Segment', 'Product Category', 'Region', 'State or Province', 'City',
                         'Product Sub-Category','Product Container', 'Product Name', 'Country'], axis=1).corr())

data['Shipping Cost'].isnull()

data.drop(['Prod'])

data[data['Row ID'] == 21776]['Profit']

def calc_Pbm(data):
    profit= data['Profit']
    sales = data['Sales']
    pbm = data['Product Base Margin']
    cost = sales - profit
    if pd.isna(pbm):
        return (profit/cost) * 100
    else:
        return pbm
    
data['Product Base Margin'] = data.apply(calc_Pbm, axis =1)

# Convert order date and ship date ito proper python objects
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])

# Extract important info and put in separate columns
data['Order_year'] = data['Order Date'].dt.year
data['Order_month'] = data['Order Date'].dt.month
data['Order_day'] = data['Order Date'].dt.day

data['Ship_year'] = data['Ship Date'].dt.year
data['Ship_month'] = data['Ship Date'].dt.month
data['Ship_day'] = data['Ship Date'].dt.day

#Calculate Order processing time Processing Time

data['Order Processing Time'] = (data['Ship Date'] - data['Order Date']).dt.days
data['Order Processing Time (Hours)'] = ((data['Ship Date'] - data['Order Date']).dt.total_seconds()) / 3600

data.drop(['Order Processing Time'], axis=1, inplace=True)
data['Order Processing Time'].value_counts()

data['Order Priority'].value_counts()
data['Order Priority'] = data['Order Priority'].replace({
    'Critical ': 'Critical'
})

data['Customer Segment'].value_counts()
data['Product Category'].value_counts()

# Creating new calculated fields
#Creating Profit Margin
data['Profit Margin'] = data[['Profit', 'Sales']].apply(lambda x: (x['Profit'] / x['Sales']) * 100, axis =1)
data.columns

# Creating Discount Impact Factor 
def create_dif(df):
    unit_price = df['Unit Price']
    quantity_ordered = df['Quantity ordered new']
    discount = df['Discount']
    Sales_before_discount = unit_price * quantity_ordered
    Sales_after_discount = Sales_before_discount * (1 - discount)

    return (Sales_before_discount - Sales_after_discount)

data['Discount Impact Factor'] = data.apply(create_dif, axis=1)

# Task 2
# Performing Exploratory Data Analysis
# Sales and Profit Analysis
#Total Sales
np.sum(data['Sales'])
# Total Profit
np.sum(data['Profit'])
# Average profit margin
np.mean(data['Profit Margin'])

# Identifying the most profitable product categories and Sub-categories
# Most profitable Product categories
data['Product Category'].value_counts()

data[data['Product Category'] == 'Office Supplies']['Profit'].sum() # Largest profit
data[data['Product Category'] == 'Technology']['Profit'].sum()
data[data['Product Category'] == 'Furniture']['Profit'].sum() # Least Profitable

# Most profitable Sub-Categories
data['Product Sub-Category'].value_counts()


data[data['Product Sub-Category'] == 'Paper']['Profit'].sum()
data[data['Product Sub-Category'] == 'Binders and Binder Accessories']['Profit'].sum() # Most Profitable
data[data['Product Sub-Category'] == 'Telephones and Communication']['Profit'].sum()
data[data['Product Sub-Category'] == 'Office Furnishings']['Profit'].sum()
data[data['Product Sub-Category'] == 'Computer Peripherals']['Profit'].sum()
data[data['Product Sub-Category'] == 'Pens & Art Supplies']['Profit'].sum() # fifth runner up loss making product
data[data['Product Sub-Category'] == 'Storage & Organization']['Profit'].sum()
data[data['Product Sub-Category'] == 'Appliances']['Profit'].sum()
data[data['Product Sub-Category'] == 'Office Machines']['Profit'].sum()
data[data['Product Sub-Category'] == 'Chairs & Chairmats']['Profit'].sum()
data[data['Product Sub-Category'] == 'Tables']['Profit'].sum() # Least profitable/ loss making product
data[data['Product Sub-Category'] == 'Labels']['Profit'].sum()
data[data['Product Sub-Category'] == 'Envelopes']['Profit'].sum() # Third runner up loss making product
data[data['Product Sub-Category'] == 'Bookcases']['Profit'].sum() # Fourth runner up loss
data[data['Product Sub-Category'] == 'Scissors, Rulers and Trimmers']['Profit'].sum() # second runner up loss making product
data[data['Product Sub-Category'] == 'Rubber Bands']['Profit'].sum() # first runner up loss making product
data[data['Product Sub-Category'] == 'Copiers and Fax']['Profit'].sum()

# Impact of Discounts on Sales and Profit
# Correlation between discount rates and profit margins
data[['Discount', 'Profit Margin']].corr() # There is a weak correlation between discount rates and profit margins

# Identifying Thresholds at which discounts become unprofitable
data.groupby('Discount')['Profit Margin'].mean()# This shows an unsteady discount/profit margin correlation

# Sales performance of discounted vs non-discounted products
data[data['Discount'] == 0.00]['Sales'].mean()
data[data['Discount'] > 0.00]['Sales'].mean()

data.groupby('Discount')['Sales'].sum().max() # The data shows that more sales were made overall in non discounted locations less than discounted. However, on average, more sales were made when no discount was offered

#Shipping vs logistics analysis
# Comparing the cost of different shipping modes
data.groupby('Ship Mode')['Shipping Cost'].sum() # It costs more using Delivery Trucks, Regular air is next, while Express Air cost the least.
data.groupby('Ship Mode')['Shipping Cost'].mean()# On average, it costs less using Regular air than Express air. However, it still costs more on average when using the delivery truck

# Fastest and Slowest delivery times
data['Order Processing Time'].max()#10 days
data['Order Processing Time'].min()#Same day
data.groupby('Order Processing Time').min()['Ship Mode']# The mode with the fastest delivery time is mostly The delivery truck but once for both Express air and Regular Air
data.groupby('Order Processing Time').max()['Ship Mode']# The mode with the slowest delivery time is always Regular Air

# Correlation between Shipping cost and profitability
data[['Shipping Cost', 'Profit']].corr()# The data shows a very lokw correlation between shipping cost and profitability

# Regional and State-wise Performance
data.groupby('Region')['Profit'].sum() # The Eastern Region is showing itself to be the topmost profitable region
data.groupby('Region')['Sales'].sum() # The East once again proves to be the region with the most sales, the South, which produced a loss provves to have the least sales

#How geographical location affects shipping cost and profit
data.groupby('Region')['Shipping Cost'].sum()
data.groupby('Region')['Shipping Cost'].mean()# Averagely, it costs more to transport to the east showing that the depots are closer to other regions
data.groupby('Ship Mode')['Order Processing Time'].mean()
data.groupby('Region')['Profit'].mean()# More profit is made in the east per product than at any other place

# Customer Behaviour and Segmentation
data.groupby('Customer Segment')['Sales'].sum()# Most sales come from the corporate customer segment

# Most price sensitive aspect
data.groupby('Customer Segment')['Discount'].mean()
data.groupby('Discount')['Customer Segment'].value_counts()
data.groupby('Discount')['Customer Segment'].mean()

df = pd.DataFrame(data.groupby('Discount')['Customer Segment'].value_counts())

sns.histplot(data=data, x='Discount', hue='Customer Segment', multiple='dodge')
plt.tight_layout()# This graph shows that the Corporate Customer Segment is the most price sensitive

#Data Visualization & Reporting
#Sales & Profit Trends
plt.figure(figsize=(10,6))
sns.lineplot(data=data, x='Ship_month', y='Sales')

#Top product categories
data.groupby('Product Category')['Profit'].sum().sort_values().plot(kind='bar')

#Top Product Sub Categories
data.groupby('Product Sub-Category')['Profit'].sum().sort_values().plot(kind='bar', )

# Discount Impact Diagram
plt.scatter(y= 'Profit', x='Discount', data=data)

# Regional Performance
pivot_table = pd.pivot_table(data, values='Sales', index='Region', columns='Product Category', aggfunc='sum')
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='coolwarm')
sns.heatmap(pd.pivot_table(data, values='Sales', index='State or Province', columns='Product Category', aggfunc='sum'), annot=True, fmt='.0f', cmap='coolwarm')

#Customer Segments
# Calculate the sum of sales by Customer Segment
segment_sales = data.groupby('Customer Segment')['Sales'].sum()

# Create the pie chart directly from the pandas Series
plt.figure(figsize=(10, 8))
segment_sales.plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    shadow=False,
    explode=[0.05] * len(segment_sales),
    wedgeprops={'edgecolor': 'white'},
    title='Revenue Contribution by Customer Segment'
)

plt.axis('equal')
plt.ylabel('')  # Remove the y-label

# If you want to add formatted labels with sales values to a legend
total = segment_sales.sum()
labels = [f"{segment} (${sales:,.0f})" for segment, sales in segment_sales.items()]
plt.legend(labels, loc='best', bbox_to_anchor=(1, 0.5))
plt.tight_layout()


#Shipping Efficiency
# Reshape the data for seaborn
ship_comparison = data.groupby('Ship Mode').agg({
    'Shipping Cost': 'mean',
    'Order Processing Time': 'mean'
}).reset_index()

ship_data_long = pd.melt(ship_comparison, 
                         id_vars=['Ship Mode'],
                         value_vars=['Shipping Cost', 'Order Processing Time'],
                         var_name='Metric', value_name='Value')

# Create a grouped bar chart with seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='Ship Mode', y='Value', hue='Metric', data=ship_data_long)

plt.title('Comparison of Shipping Costs and Delivery Times by Shipping Mode', fontsize=14)
plt.xlabel('Shipping Mode', fontsize=12)
plt.ylabel('Value ($ / Days)', fontsize=12)
plt.legend(title='')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

data.groupby('Shipping Cost')['Region']
data.groupby('Product Category')['Customer Segment'].sum()




# Define numerical columns for correlation analysis
numerical_cols = ['Discount', 'Unit Price', 'Shipping Cost', 'Profit', 
                  'Quantity ordered new', 'Sales', 'Product Base Margin']
# Create a figure for correlation matrices
plt.figure(figsize=(20, 15))
plt.suptitle('Correlation Matrices by Customer Segment', fontsize=16)

# Define customer segments to analyze
segments = ['Corporate', 'Small Business', 'Home Office', 'Consumer']

# Create correlation matrices for each segment
for i, segment in enumerate(segments, 1):
    plt.subplot(2, 2, i)
    
    # Filter data for the current segment
    segment_data = data[data['Customer Segment'] == segment][numerical_cols]
    
    # Calculate correlation matrix
    corr_matrix = segment_data.corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                fmt='.2f', linewidths=0.5)
    plt.title(f'{segment} Segment Correlation Matrix')
    plt.tight_layout()

plt.subplots_adjust(top=0.92)
plt.savefig('segment_correlation_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

chlor_sales = data.groupby('State or Province')['Sales'].sum()
len(chlor_sales.values)
chlor_states = data['State or Province']
states = sorted(chlor_states.unique())
len(states)

# Geographical Plotting 
import plotly.express as px
import pandas as pd

# Sample sales data with full state names
dat = {
    'state': states,
    'sales': chlor_sales.values
}
df = pd.DataFrame(dat)
df
# 🔍 Make sure 'sales' is numeric
df['sales'] = pd.to_numeric(df['sales'], errors='coerce')

# 🔍 Use state abbreviations for correct mapping
state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DoC'
}

# Convert full names to abbreviations
df['state_code'] = df['state'].map(state_abbrev)
df['sales'] = np.rint(df['sales']).astype(int)
# 🌍 Plot the corrected choropleth
fig = px.choropleth(
    df,
    locations='state_code',
    locationmode='USA-states',
    color='sales',
    color_continuous_scale='Blues',
    scope='usa',
    labels={'sales': 'Sales ($)'},
    title='Sales by U.S. State'
)

fig.update_layout(geo=dict(showlakes=True, lakecolor='white'))
fig.show()
df


