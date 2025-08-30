# Financial AI Assistant using IBM Granite 3.2-2B
# Complete implementation for Google Colab with Gradio interface

# Install required packages
!pip install transformers torch gradio accelerate bitsandbytes scipy yfinance matplotlib

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import random


class FinancialAIAssistant:
    def __init__(self):
        self.model_name = "ibm-granite/granite-3.2-2b-instruct"
        self.tokenizer = None
        self.model = None
        self.user_profiles = {}
        self.conversation_history = {}

        # Load model with optimized settings for Colab
        self.load_model()

        # Financial knowledge base - Updated tips
        self.financial_tips = {
            "savings": {
                "student": "Start with the 50/30/20 rule: 50% needs, 30% wants, 20% savings. Even $25/month builds good habits!",
                "professional": "Aim for 6-month emergency fund, then maximize employer 401(k) match before other investments.",
                "family": "Consider 529 plans for children's education and increase life insurance coverage."
            },
            "investment": {
                "student": "Begin with low-cost index funds and robo-advisors. Start small but start early!",
                "professional": "Diversify across asset classes. Consider tax-advantaged accounts like IRA/401(k) first.",
                "family": "Balance growth and stability. Consider target-date funds and estate planning."
            },
            "budgeting": {
                "student": "Track spending with apps, cook at home, and look for student discounts everywhere.",
                "professional": "Automate savings, review subscriptions monthly, and negotiate salary regularly.",
                "family": "Plan for irregular expenses, use family budgeting apps, and involve older children in financial discussions."
            },
            # New tips related to new features
            "spending_analysis": {
                "general": "Review your spending visualization regularly to identify areas where you can cut back. Small changes add up!",
                "student": "After analyzing spending, look for student-specific discounts or free alternatives for entertainment and food.",
                "professional": "Use spending insights to find 'leaky' subscriptions or impulse purchases that hinder savings.",
                "family": "Analyze family spending together to teach children about financial responsibility and identify cost-saving opportunities."
            },
             "stock_data": {
                 "general": "Looking at stock data is a great first step! Remember that past performance doesn't guarantee future results. Research the company and understand its fundamentals.",
                 "student": "Start by tracking stocks in industries you understand or use daily. Consider paper trading before using real money.",
                 "professional": "Use stock data to inform your diversified investment strategy, but don't make impulsive decisions based on daily price swings.",
                 "family": "If investing for long-term goals like retirement or education, focus on the overall trend and dividend history rather than short-term price changes."
             },
             "financial_games": {
                 "general": "Financial games are a fun way to learn! Apply the lessons from the games to your real-life financial decisions.",
                 "budgeting_challenge": "Did you play the Budgeting Challenge? Think about which expenses were the hardest to manage and how you could plan better next time.",
                 "investment_simulation": "Played the Investment Simulation? Remember that diversification and long-term perspective are key, even when the market is volatile."
             }
        }


    def load_model(self):
        """Load IBM Granite model with memory optimization for Colab"""
        print("Loading IBM Granite 3.2-2B model...")

        # Configure for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU mode...")
            # Fallback for limited resources
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

    def identify_user_type(self, profile_info):
        """Determine user type based on profile information"""
        profile_lower = profile_info.lower()

        if any(word in profile_lower for word in ['student', 'college', 'university', 'studying']):
            return 'student'
        elif any(word in profile_lower for word in ['family', 'married', 'children', 'kids', 'parent']):
            return 'family'
        else:
            return 'professional'

    def adjust_communication_style(self, user_type, response):
        """Adjust response based on user demographics"""
        # This function is currently a placeholder as style adjustment is handled by the prompt
        return response

    def analyze_spending_pattern(self, expenses_text):
        """Analyze spending patterns and provide insights"""
        # Parse expenses from text input
        expenses = {}
        lines = expenses_text.strip().split('\n')

        for line in lines:
            # Try to parse "Category: Amount" format
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    category = parts[0].strip()
                    try:
                        amount = float(re.findall(r'\d+\.?\d*', parts[1])[0])
                        if category in expenses:
                            expenses[category] += amount
                        else:
                            expenses[category] = amount
                    except:
                        continue

        if not expenses:
            return "Please provide expenses in format 'Category: Amount' (one per line)"

        # Calculate insights
        total = sum(expenses.values())
        insights = []

        # High spending categories
        high_spending = [(cat, amt) for cat, amt in expenses.items() if total > 0 and amt/total > 0.3]
        if high_spending:
            insights.append(f"⚠️ High spending detected: {high_spending[0][0]} accounts for {high_spending[0][1]/total*100:.1f}% of total")

        # Suggestions based on categories
        if any(cat.lower() in ['dining out', 'restaurants'] for cat in expenses.keys()):
            insights.append("💡 Consider meal prep to reduce dining out costs.")

        if any(cat.lower() in ['subscriptions', 'memberships'] for cat in expenses.keys()):
            insights.append("💡 Review and cancel unused subscriptions.")

        return "\n".join(insights) if insights else "Spending looks well-distributed!"

    def generate_budget_summary(self, income, expenses_dict):
        """Generate comprehensive budget summary"""
        try:
            income = float(income)
            total_expenses = sum(float(v) for v in expenses_dict.values() if v)
            savings = income - total_expenses
            savings_rate = (savings / income) * 100 if income > 0 else 0

            summary = f"""
📊 BUDGET SUMMARY
═══════════════════
Monthly Income: ${income:,.2f}
Total Expenses: ${total_expenses:,.2f}
Net Savings: ${savings:,.2f}
Savings Rate: {savings_rate:.1f}%

📈 FINANCIAL HEALTH SCORE
{self.calculate_health_score(savings_rate)}

💰 EXPENSE BREAKDOWN
"""
            for category, amount in expenses_dict.items():
                if amount:
                    percentage = (float(amount) / total_expenses) * 100
                    summary += f"• {category}: ${float(amount):,.2f} ({percentage:.1f}%)\n"

            return summary

        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def calculate_health_score(self, savings_rate):
        """Calculate financial health score"""
        if savings_rate >= 20:
            return "🟢 EXCELLENT (20%+ savings rate)"
        elif savings_rate >= 10:
            return "🟡 GOOD (10-20% savings rate)"
        elif savings_rate >= 0:
            return "🟠 NEEDS IMPROVEMENT (0-10% savings rate)"
        else:
            return "🔴 CRITICAL (Spending exceeds income!)"


    def generate_response(self, user_input, user_profile, conversation_context="", feature_context=""):
        """Generate AI response using IBM Granite model"""
        user_type = self.identify_user_type(user_profile)

        # Build context-aware prompt
        system_prompt = f"""You are a helpful financial advisor AI assistant.

User Profile: {user_profile}
User Type: {user_type}
Communication Style: Adjust your tone and complexity for a {user_type}.

Previous Conversation Context: {conversation_context}

Recent Feature Usage: {feature_context}

Consider the user's previous conversation and their recent interactions with features like budget analysis, spending visualization, stock data lookups, financial games, net worth tracking, or loan calculations when generating your response.

Provide personalized financial advice that is:
1. Tailored to their demographic and situation
2. Practical and actionable
3. Educational but not overwhelming
4. Encouraging and supportive

User Question: {user_input}

Response:"""

        try:
            # Tokenize and generate
            inputs = self.tokenizer(
                system_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )

            # Move to appropriate device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the new response part
            response = full_response[len(system_prompt):].strip()

            # Add demographic-specific and feature-specific tips
            added_tip = False
            if user_type in self.financial_tips:
                for category, tips in self.financial_tips[user_type].items():
                    if category.lower() in user_input.lower() and not added_tip:
                        response += f"\n\n💡 Quick tip for {user_type}s: {tips}"
                        added_tip = True
                        break # Only add one user-type specific tip initially

            # Add general tips based on recent feature usage if no specific tip added
            if not added_tip and feature_context:
                 for feature_category, tips_dict in self.financial_tips.items():
                     if feature_category in ["spending_analysis", "stock_data", "financial_games", "net_worth", "loan_payment"] and feature_category.replace('_', ' ') in feature_context.lower():
                         general_tip = tips_dict.get("general")
                         if general_tip and not added_tip:
                             response += f"\n\n💡 Quick tip related to {feature_category.replace('_', ' ')}: {general_tip}"
                             added_tip = True
                             break
                         # Add specific game tips if applicable
                         if feature_category == "financial_games":
                             if "budgeting challenge" in feature_context.lower():
                                 game_tip = tips_dict.get("budgeting_challenge")
                                 if game_tip and not added_tip:
                                     response += f"\n\n💡 Tip from the game: {game_tip}"
                                     added_tip = True
                                     break
                             elif "investment simulation" in feature_context.lower():
                                 game_tip = tips_dict.get("investment_simulation")
                                 if game_tip and not added_tip:
                                     response += f"\n\n💡 Tip from the game: {game_tip}"
                                     added_tip = True
                                     break


            return response

        except Exception as e:
            return f"I apologize, but I encountered an error generating a response: {str(e)}. Please try rephrasing your question."

# Initialize the AI assistant
ai_assistant = FinancialAIAssistant()

def chat_interface(message, profile, history, recent_feature_usage=""):
    """Main chat interface function"""
    if not message.strip():
        return history, history, ""

    # Get AI response
    context = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-3:]]) if history else ""
    response = ai_assistant.generate_response(message, profile, conversation_context=context, feature_context=recent_feature_usage)

    # Update history
    history.append([message, response])

    # Clear recent feature usage after it's been incorporated into the response
    return history, history, "", "" # Return empty string for recent_feature_usage state

def budget_analysis(income, housing, food, transport, entertainment, other, profile):
    """Budget analysis functionality"""
    if not income:
        return "Please enter your monthly income to generate a budget summary.", ""

    expenses = {
        'Housing': housing or 0,
        'Food': food or 0,
        'Transportation': transport or 0,
        'Entertainment': entertainment or 0,
        'Other': other or 0
    }

    summary = ai_assistant.generate_budget_summary(income, expenses)

    # Add personalized recommendations
    user_type = ai_assistant.identify_user_type(profile)
    recommendations = f"\n\n🎯 PERSONALIZED RECOMMENDATIONS FOR {user_type.upper()}S:\n"

    if user_type == 'student':
        recommendations += "• Focus on building emergency fund first\n• Look for student discounts on everything\n• Consider part-time work or internships"
    elif user_type == 'professional':
        recommendations += "• Maximize employer benefits and 401(k) matching\n• Consider tax-loss harvesting\n• Review insurance coverage annually"
    else:  # family
        recommendations += "• Start children's education fund (529 plan)\n• Review life insurance needs\n• Plan for family emergencies and medical expenses"

    # Capture feature usage details for context
    feature_details = f"User recently performed a Budget Analysis with Income: ${income}, Expenses: {sum(expenses.values()):,.2f}. Summary: {summary.splitlines()[5].strip()}."
    return summary + recommendations, feature_details # Return feature details along with summary

def spending_insights(expenses_input, profile):
    """Analyze spending patterns"""
    if not expenses_input.strip():
        return "Please enter your expenses in the format 'Category: Amount' (one per line)", ""

    insights = ai_assistant.analyze_spending_pattern(expenses_input)
    user_type = ai_assistant.identify_user_type(profile)

    # Add AI-generated insights
    ai_prompt = f"Analyze these spending patterns for a {user_type}: {expenses_input}. Provide 3 specific, actionable recommendations."
    ai_insights = ai_assistant.generate_response(ai_prompt, profile)

    # Capture feature usage details for context
    feature_details = f"User recently used Spending Insights with expenses: {expenses_input.replace('\n', ', ')}. Insights: {insights.splitlines()[0] if insights else 'No major insights.'}"
    return f"{insights}\n\n🤖 AI INSIGHTS:\n{ai_insights}", feature_details # Return feature details

def plot_spending(expenses_text):
    """Generate a pie chart visualizing expense breakdown."""
    expenses = {}
    lines = expenses_text.strip().split('\n')

    for line in lines:
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                category = parts[0].strip()
                try:
                    amount = float(re.findall(r'\d+\.?\d*', parts[1])[0])
                    if category in expenses:
                      expenses[category] += amount
                    else:
                      expenses[category] = amount
                except:
                    continue

    if not expenses:
        return None # Return None if no valid expenses are provided

    labels = expenses.keys()
    sizes = expenses.values()
    colors = plt.cm.Paired(np.arange(len(labels))) # Use a colormap

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title("Spending Breakdown by Category")
    return fig

def plot_spending_with_context(expenses_text):
    """Generate a pie chart visualizing expense breakdown and return context."""
    plot = plot_spending(expenses_text) # Call the original plotting function
    if plot is None:
        return None, "" # Return None plot and empty context if no data

    # Capture feature usage details for context
    feature_details = f"User recently generated a Spending Visualization plot with expenses input: {expenses_text.replace('\n', ', ')}. Categories plotted."
    return plot, feature_details # Return plot and feature details


def get_stock_details(symbol):
    """Fetches and formats real-time stock details."""
    if not symbol:
        return "Please enter a stock symbol.", ""

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Check if the ticker is valid and has data
        if not info or 'regularMarketPrice' not in info:
             return f"Could not retrieve data for symbol: {symbol}. Please check the symbol.", ""

        details = {
            "Symbol": info.get('symbol', 'N/A'),
            "Current Price": info.get('regularMarketPrice', 'N/A'),
            "Open": info.get('regularMarketOpen', 'N/A'),
            "High": info.get('regularMarketDayHigh', 'N/A'),
            "Low": info.get('regularMarketDayLow', 'N/A'),
            "Previous Close": info.get('previousClose', 'N/A'),
            "Volume": info.get('regularMarketVolume', 'N/A'),
            "Market Cap": info.get('marketCap', 'N/A')
        }

        formatted_output = "📊 Stock Details:\n"
        for key, value in details.items():
            if isinstance(value, (int, float)) and key not in ["Volume", "Market Cap"]:
                 formatted_output += f"{key}: ${value:,.2f}\n"
            elif isinstance(value, (int, float)) and key in ["Volume", "Market Cap"]:
                 formatted_output += f"{key}: {value:,}\n"
            else:
                 formatted_output += f"{key}: {value}\n"

        # Capture feature usage details for context
        feature_details = f"User recently looked up stock details for {symbol}. Current Price: ${details['Current Price']}."
        return formatted_output, feature_details # Return feature details

    except Exception as e:
        return f"An error occurred while fetching stock data for {symbol}: {str(e)}", ""

def investment_advice(risk_tolerance, investment_amount, time_horizon, profile):
    """Provide personalized investment advice"""
    user_type = ai_assistant.identify_user_type(profile)

    prompt = f"""
    Provide investment advice for a {user_type} with:
    - Risk tolerance: {risk_tolerance}
    - Investment amount: ${investment_amount}
    - Time horizon: {time_horizon}

    Include specific fund suggestions, allocation percentages, and platform recommendations.
    """

    response = ai_assistant.generate_response(prompt, profile)

    # Capture feature usage details for context
    feature_details = f"User recently requested Investment Advice with Risk: {risk_tolerance}, Amount: ${investment_amount}, Horizon: {time_horizon}."
    return response, feature_details # Return feature details

def tax_optimization(income_range, deductions, profile):
    """Provide tax optimization strategies"""
    user_type = ai_assistant.identify_user_type(profile)

    prompt = f"""
    Provide tax optimization strategies for a {user_type} with:
    - Income range: {income_range}
    - Current deductions: {deductions}

    Include specific strategies, deadlines, and potential savings.
    """

    response = ai_assistant.generate_response(prompt, profile)

    # Capture feature usage details for context
    feature_details = f"User recently requested Tax Optimization advice for Income range: {income_range}, Deductions: {deductions[:50]}..."
    return response, feature_details # Return feature details

def budgeting_challenge(user_profile):
    """A simple text-based budgeting challenge game."""
    user_type = ai_assistant.identify_user_type(user_profile)
    starting_cash = 1000
    weeks = 4
    expenses_options = {
        "Food": random.randint(50, 150),
        "Transportation": random.randint(30, 100),
        "Entertainment": random.randint(20, 80),
        "Other": random.randint(10, 50)
    }
    income_options = {
        "Part-time Job": random.randint(100, 300),
        "Allowance": random.randint(50, 100),
        "Freelance Gig": random.randint(75, 250)
    }

    output = f"Welcome to the Budgeting Challenge for {user_type}s!\n"
    output += f"You start with ${starting_cash} and have {weeks} weeks to manage your money.\n\n"

    current_cash = starting_cash
    total_income = 0
    total_expenses = 0

    for week in range(1, weeks + 1):
        output += f"--- Week {week} ---\n"
        weekly_income = random.choice(list(income_options.values())) if random.random() > 0.3 else 0 # Chance of no income
        weekly_expenses = {cat: amount for cat, amount in expenses_options.items()} # Simplified: fixed weekly expenses

        output += f"Income this week: ${weekly_income}\n"
        output += "Expenses this week:\n"
        for cat, amount in weekly_expenses.items():
            output += f"- {cat}: ${amount}\n"

        current_cash += weekly_income - sum(weekly_expenses.values())
        total_income += weekly_income
        total_expenses += sum(weekly_expenses.values())

        output += f"Cash at the end of week {week}: ${current_cash}\n\n"

    output += "--- Challenge Summary ---\n"
    output += f"Total Income: ${total_income}\n"
    output += f"Total Expenses: ${total_expenses}\n"
    output += f"Ending Cash: ${current_cash}\n\n"

    if current_cash >= starting_cash:
        output += "🎉 Congratulations! You ended with more money than you started. Great budgeting!\n"
    elif current_cash >= 0:
        output += "👍 You stayed within your budget! Good job managing your finances.\n"
    else:
        output += "😟 You ended with negative cash. Consider reviewing your spending habits.\n"

    # Capture feature usage details for context
    feature_details = f"User recently played the Budgeting Challenge. Ending Cash: ${current_cash:,.2f}."
    return output, feature_details # Return feature details

def investment_simulation(user_profile):
    """A simple text-based investment simulation game."""
    user_type = ai_assistant.identify_user_type(user_profile)
    starting_portfolio_value = 5000
    months = 6
    simulated_returns = [random.uniform(-0.05, 0.10) for _ in range(months)] # Simulate monthly returns

    output = f"Welcome to the Investment Simulation for {user_type}s!\n"
    output += f"You start with a portfolio value of ${starting_portfolio_value:,.2f} and simulate {months} months of market activity.\n\n"

    current_portfolio_value = starting_portfolio_value

    for month in range(1, months + 1):
        monthly_return = simulated_returns[month-1]
        return_amount = current_portfolio_value * monthly_return
        current_portfolio_value += return_amount

        output += f"--- Month {month} ---\n"
        output += f"Monthly Return: {monthly_return:.2%}\n"
        output += f"Return Amount: ${return_amount:,.2f}\n"
        output += f"Portfolio Value at end of month {month}: ${current_portfolio_value:,.2f}\n\n"

    output += "--- Simulation Summary ---\n"
    output += f"Starting Portfolio Value: ${starting_portfolio_value:,.2f}\n"
    output += f"Ending Portfolio Value: ${current_portfolio_value:,.2f}\n\n"

    if current_portfolio_value >= starting_portfolio_value:
        output += "📈 Your portfolio grew! This simulation shows the power of investing over time.\n"
    else:
        output += "📉 Your portfolio value decreased in this simulation. Remember that investments carry risk and values can fluctuate.\n"

    # Capture feature usage details for context
    feature_details = f"User recently played the Investment Simulation. Ending Portfolio Value: ${current_portfolio_value:,.2f}."
    return output, feature_details # Return feature details

def start_selected_game_with_context(game_name, profile):
    """Starts the selected financial game and returns context."""
    if game_name == "Budgeting Challenge":
        game_result, feature_details = budgeting_challenge(profile)
    elif game_name == "Investment Simulation":
        game_result, feature_details = investment_simulation(profile)
    else:
        game_result = "Please select a game to start."
        feature_details = "" # Return empty feature details for no game

    return game_result, feature_details

# Add Net Worth Tracker function
def calculate_net_worth(assets_text, liabilities_text):
    """Calculates net worth based on assets and liabilities."""
    assets = {}
    liabilities = {}
    total_assets = 0
    total_liabilities = 0

    # Parse assets
    for line in assets_text.strip().split('\n'):
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                category = parts[0].strip()
                try:
                    amount = float(re.findall(r'\d+\.?\d*', parts[1])[0])
                    if category in assets:
                        assets[category] += amount
                    else:
                        assets[category] = amount
                except:
                    continue

    # Parse liabilities
    for line in liabilities_text.strip().split('\n'):
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                category = parts[0].strip()
                try:
                    amount = float(re.findall(r'\d+\.?\d*', parts[1])[0])
                    if category in liabilities:
                        liabilities[category] += amount
                    else:
                        liabilities[category] = amount
                except:
                    continue

    total_assets = sum(assets.values())
    total_liabilities = sum(liabilities.values())
    net_worth = total_assets - total_liabilities

    output = "📊 Net Worth Calculation:\n"
    output += "════════════════════════\n"
    output += f"Total Assets: ${total_assets:,.2f}\n"
    output += f"Total Liabilities: ${total_liabilities:,.2f}\n"
    output += f"Your Net Worth: ${net_worth:,.2f}\n\n"

    if net_worth >= 0:
        output += "🟢 Your net worth is positive! Keep building your assets and reducing liabilities.\n"
    else:
        output += "🟠 Your liabilities exceed your assets. Focus on paying down debt and increasing savings.\n"

    # Capture feature usage details for context
    feature_details = f"User recently calculated Net Worth. Total Assets: ${total_assets:,.2f}, Total Liabilities: ${total_liabilities:,.2f}, Net Worth: ${net_worth:,.2f}."
    return output, feature_details

# Add Loan Payment Calculator function
def calculate_loan_payment(principal, annual_interest_rate, years):
    """Calculates the monthly payment for a loan."""
    try:
        principal = float(principal)
        annual_interest_rate = float(annual_interest_rate)
        years = int(years)

        if principal <= 0 or annual_interest_rate < 0 or years <= 0:
            return "Please enter positive values for principal, interest rate, and years.", ""

        monthly_interest_rate = (annual_interest_rate / 100) / 12
        number_of_payments = years * 12

        if monthly_interest_rate == 0:
            monthly_payment = principal / number_of_payments
        else:
            # M = P [ i(1 + i)^n ] / [ (1 + i)^n – 1]
            monthly_payment = principal * (monthly_interest_rate * (1 + monthly_interest_rate)**number_of_payments) / ((1 + monthly_interest_rate)**number_of_payments - 1)

        output = "📈 Loan Payment Calculation:\n"
        output += "═════════════════════════\n"
        output += f"Principal Loan Amount: ${principal:,.2f}\n"
        output += f"Annual Interest Rate: {annual_interest_rate:.2f}%\n"
        output += f"Loan Term: {years} years\n"
        output += f"Estimated Monthly Payment: ${monthly_payment:,.2f}\n\n"

        output += "💡 Consider making extra payments to reduce the total interest paid and pay off the loan faster."

        # Capture feature usage details for context
        feature_details = f"User recently calculated a Loan Payment. Principal: ${principal:,.2f}, Rate: {annual_interest_rate:.2f}%, Term: {years} years. Monthly Payment: ${monthly_payment:,.2f}."
        return output, feature_details

    except ValueError:
        return "Please ensure all inputs are valid numbers.", ""
    except Exception as e:
        return f"An error occurred during calculation: {str(e)}", ""

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""

    with gr.Blocks(
        title="Financial AI Assistant - Powered by IBM Granite 3.2-2B",
        theme=gr.themes.Soft(),
        css="""
        /* General styles */
        .gradio-container {
            font-family: 'Roboto', sans-serif; /* Use a common, modern font */
            background: linear-gradient(to bottom right, #e0f7fa, #b2ebf2); /* Light gradient background */
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        /* Header and Logo */
        .header-row {
            align-items: center; /* Vertically align logo and title */
            margin-bottom: 30px;
        }
        .logo-col {
            display: flex;
            justify-content: center; /* Center logo horizontally */
            align-items: center; /* Center logo vertically */
        }
        .logo img {
            max-width: 180px; /* Slightly larger logo */
            height: auto;
            border-radius: 10px; /* Slightly rounded corners for logo */
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .title-col {
            text-align: left; /* Align title to the left */
        }
        .title-col h1 {
            color: #00796b; /* Teal color for main title */
            margin-bottom: 5px;
        }
         .title-col h3 {
            color: #009688; /* Slightly lighter teal for subtitle */
            margin-top: 0;
         }


        /* Profile Section */
        .gradio-container > div > div:nth-child(2) { /* Target the row below header */
            margin-bottom: 20px;
        }

        /* Tabs */
        .gradio-tabs {
            border-radius: 10px;
            overflow: hidden; /* Ensures border-radius applies to tabs */
        }
        .gradio-tabs > div[role="tablist"] {
            background-color: #e0f2f7; /* Light blue background for tab headers */
            border-bottom: 2px solid #0097a7; /* Border below tab headers */
        }
        .gradio-tabs > div[role="tablist"] > button {
            color: #004d40; /* Dark teal color for inactive tab text */
            font-weight: normal;
        }
        .gradio-tabs > div[role="tablist"] > button.selected {
            color: #00796b; /* Teal color for active tab text */
            border-bottom: 2px solid #00796b; /* Highlight active tab */
            font-weight: bold;
            background-color: #ffffff; /* White background for active tab */
        }


        /* Chatbot */
        .chat-message {
            padding: 12px;
            margin: 8px;
            border-radius: 18px; /* More rounded corners */
            max_width: 80%; /* Limit message width */
        }
        .chat-message.user {
            background-color: #e0f7fa; /* Light cyan for user messages */
            align_self: flex-end; /* Align user messages to the right */
        }
        .chat-message.bot {
            background-color: #b2ebf2; /* Lighter cyan for bot messages */
            align_self: flex-start; /* Align bot messages to the left */
        }
         .chat-message p {
             margin: 0; /* Remove default paragraph margin */
         }


        /* Buttons */
        button.primary {
            background-color: #00796b; /* Teal primary button */
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 1em;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        button.primary:hover {
            background-color: #004d40; /* Darker teal on hover */
        }
         button.secondary {
            background-color: #80cbc4; /* Light teal secondary button */
            color: #004d40;
            border: 1px solid #00796b;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 1em;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        button.secondary:hover {
            background-color: #e0f2f7; /* Lighter background on hover */
        }

        /* Textboxes and Inputs */
        input[type="text"], textarea, input[type="number"] {
            border: 1px solid #b2ebf2; /* Light border */
            border-radius: 8px;
            padding: 10px;
            font-size: 1em;
        }
         label {
             font-weight: bold;
             color: #004d40; /* Dark teal for labels */
         }

        /* Footer */
        .gradio-container > div:last-child {
            margin-top: 30px;
            text-align: center;
            color: #555;
            font-size: 0.9em;
        }

        """
    ) as interface:

        # Header with Logo and Title
        with gr.Row(equal_height=True, elem_classes="header-row"):
            with gr.Column(scale=1, elem_classes="logo-col"):
                # Actual Logo Image
                gr.Image(
                    value="https://raw.githubusercontent.com/gradio-app/gradio/main/guides/assets/logo.png", # Replace with actual logo path/URL
                    label="Financial AI Assistant Logo",
                    show_label=False,
                    container=False,
                    elem_classes="logo"
                )
            with gr.Column(scale=4, elem_classes="title-col"):
                gr.Markdown("""
                # 💰 Financial AI Assistant
                ### Powered by IBM Granite 3.2-2B Instruct Model

                Get personalized financial guidance, budget analysis, and investment advice tailored to your profile!
                """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 👤 Your Profile")
                user_profile = gr.Textbox(
                    label="Tell us about yourself",
                    placeholder="e.g., 22-year-old college student majoring in Computer Science, part-time job, living with roommates...",
                    lines=3,
                    value=""
                )

        # State variable to store recent feature usage
        recent_feature_usage = gr.State("")

        with gr.Tabs(elem_classes="gradio-tabs"):
            # Tab 1: Conversational Chat (Existing)
            with gr.TabItem("💬 Financial Chat"):
                gr.Markdown("### Ask any financial question and get personalized advice!")

                chatbot = gr.Chatbot(
                    label="Financial Advisor Chat",
                    height=400,
                    bubble_full_width=False # Note: bubble_full_width is deprecated
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask about investments, savings, budgeting, taxes, etc.",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send 💬", scale=1, variant="primary")

                # Chat functionality
                chat_history = gr.State([])

                def respond(message, profile, history, feature_usage_state):
                    # Pass the content of the state variable and clear it after use
                    response = chat_interface(message, profile, history, recent_feature_usage=feature_usage_state)
                    return response[0], response[1], response[2], "" # Return chat history, chat history, clear message, and clear feature state

                send_btn.click(
                    respond,
                    inputs=[msg, user_profile, chat_history, recent_feature_usage],
                    outputs=[chatbot, chat_history, msg, recent_feature_usage] # Outputting to clear the state
                )

                msg.submit(
                    respond,
                    inputs=[msg, user_profile, chat_history, recent_feature_usage],
                    outputs=[chatbot, chat_history, msg, recent_feature_usage] # Outputting to clear the state
                )

            # Tab 2: Budget Analysis (Existing)
            with gr.TabItem("📊 Budget Analysis"):
                gr.Markdown("### Generate detailed budget summaries and recommendations")

                with gr.Row():
                    with gr.Column():
                        monthly_income = gr.Number(
                            label="Monthly Income ($)",
                            placeholder="5000",
                            value=0
                        )

                    with gr.Column():
                        housing_exp = gr.Number(
                            label="Housing/Rent ($)",
                            placeholder="1500",
                            value=0
                        )

                with gr.Row():
                    with gr.Column():
                        food_exp = gr.Number(
                            label="Food & Groceries ($)",
                            placeholder="400",
                            value=0
                        )

                    with gr.Column():
                        transport_exp = gr.Number(
                            label="Transportation ($)",
                            placeholder="300",
                            value=0
                        )

                with gr.Row():
                    with gr.Column():
                        entertainment_exp = gr.Number(
                            label="Entertainment ($)",
                            placeholder="200",
                            value=0
                        )

                    with gr.Column():
                        other_exp = gr.Number(
                            label="Other Expenses ($)",
                            placeholder="300",
                            value=0
                        )

                budget_btn = gr.Button("Generate Budget Analysis 📊", variant="primary")
                budget_output = gr.Textbox(
                    label="Budget Analysis & Recommendations",
                    lines=15,
                    max_lines=20,
                    interactive=False # Make output non-editable
                )

                # Update the click function to capture feature details and update state
                budget_btn.click(
                    budget_analysis,
                    inputs=[monthly_income, housing_exp, food_exp, transport_exp, entertainment_exp, other_exp, user_profile],
                    outputs=[budget_output, recent_feature_usage] # Outputting the feature details to the state
                )

            # Tab 3: Spending Insights (Existing)
            with gr.TabItem("🔍 Spending Insights"):
                gr.Markdown("### Get insights on your spending habits")

                spending_input = gr.Textbox(
                    label="Enter your expenses (format: 'Category: Amount' per line)",
                    placeholder="""Groceries: 400
Dining out: 300
Gas: 150
Entertainment: 200
Subscriptions: 50""",
                    lines=8
                )

                analyze_btn = gr.Button("Analyze Spending 🔍", variant="primary")
                insights_output = gr.Textbox(
                    label="Spending Analysis & Recommendations",
                    lines=10,
                    interactive=False # Make output non-editable
                )

                # Update the click function to capture feature details and update state
                analyze_btn.click(
                    spending_insights,
                    inputs=[spending_input, user_profile],
                    outputs=[insights_output, recent_feature_usage] # Outputting the feature details to the state
                )


            # Tab 4: Investment Advisor (Existing)
            with gr.TabItem("📈 Investment Advice"):
                gr.Markdown("### Get personalized investment recommendations")

                with gr.Row():
                    risk_tolerance = gr.Radio(
                        choices=["Conservative", "Moderate", "Aggressive"],
                        label="Risk Tolerance",
                        value="Moderate"
                    )

                    investment_amount = gr.Number(
                        label="Investment Amount ($)",
                        placeholder="10000",
                        value=0
                    )

                time_horizon = gr.Radio(
                    choices=["Less than 1 year", "1-5 years", "5-10 years", "10+ years"],
                    label="Investment Time Horizon",
                    value="5-10 years"
                )

                invest_btn = gr.Button("Get Investment Advice 📈", variant="primary")
                investment_output = gr.Textbox(
                    label="Personalized Investment Recommendations",
                    lines=10,
                    interactive=False # Make output non-editable
                )

                # Update the click function to capture feature details and update state
                invest_btn.click(
                    investment_advice,
                    inputs=[risk_tolerance, investment_amount, time_horizon, user_profile],
                    outputs=[investment_output, recent_feature_usage] # Outputting the feature details to the state
                )


            # Tab 5: Tax Optimization (Existing)
            with gr.TabItem("📋 Tax Planning"):
                gr.Markdown("### Optimize your tax strategy")

                income_range = gr.Radio(
                    choices=["Under $30k", "$30k-$60k", "$60k-$100k", "$100k-$200k", "Over $200k"],
                    label="Annual Income Range",
                    value="$60k-$100k"
                )

                current_deductions = gr.Textbox(
                    label="Current Deductions/Tax Strategies",
                    placeholder="401(k) contributions, mortgage interest, charitable donations...",
                    lines=3
                )

                tax_btn = gr.Button("Get Tax Optimization Tips 📋", variant="primary")
                tax_output = gr.Textbox(
                    label="Tax Optimization Strategies",
                    lines=10,
                    interactive=False # Make output non-editable
                )

                # Update the click function to capture feature details and update state
                tax_btn.click(
                    tax_optimization,
                    inputs=[income_range, current_deductions, user_profile],
                    outputs=[tax_output, recent_feature_usage] # Outputting the feature details to the state
                )

            # New Tab: Spending Visualization
            with gr.TabItem("📊 Spending Visualization"):
                gr.Markdown("### Visualize your spending patterns")
                spending_vis_input = gr.Textbox(
                    label="Enter your expenses (format: 'Category: Amount' per line)",
                    placeholder="""Groceries: 400
Dining out: 300
Gas: 150
Entertainment: 200
Subscriptions: 50""",
                    lines=8
                )
                plot_btn = gr.Button("Generate Spending Plot 📊", variant="primary")
                spending_plot_output = gr.Plot(label="Spending Breakdown")


                plot_btn.click(
                    plot_spending_with_context,
                    inputs=[spending_vis_input],
                    outputs=[spending_plot_output, recent_feature_usage] # Outputting the feature details to the state
                )


            # New Tab: Real-time Stock Data
            with gr.TabItem("📈 Real-time Stock Data"):
                gr.Markdown("### Get real-time stock information")
                with gr.Row():
                    stock_symbol_input = gr.Textbox(label="Enter Stock Symbol", placeholder="e.g., AAPL, GOOGL")
                    get_stock_btn = gr.Button("Get Stock Details 📈", variant="primary")
                stock_output = gr.Textbox(label="Stock Details", lines=8, interactive=False) # Increased lines for more details

                # Update the click function to capture feature details and update state
                get_stock_btn.click(
                    get_stock_details,
                    inputs=[stock_symbol_input],
                    outputs=[stock_output, recent_feature_usage] # Outputting the feature details to the state
                )

            # New Tab: Net Worth Tracker
            with gr.TabItem("💰 Net Worth Tracker"):
                gr.Markdown("### Calculate your net worth")
                with gr.Row():
                    assets_input = gr.Textbox(
                        label="Enter your Assets (format: 'Category: Amount' per line)",
                        placeholder="""Cash: 5000
Investments: 10000
Car Value: 15000
Real Estate: 200000""",
                        lines=6
                    )
                    liabilities_input = gr.Textbox(
                        label="Enter your Liabilities (format: 'Category: Amount' per line)",
                        placeholder="""Credit Card Debt: 2000
Student Loans: 15000
Mortgage: 150000""",
                        lines=6
                    )
                net_worth_btn = gr.Button("Calculate Net Worth 💰", variant="primary")
                net_worth_output = gr.Textbox(
                    label="Net Worth Summary",
                    lines=8,
                    interactive=False
                )

                net_worth_btn.click(
                    calculate_net_worth,
                    inputs=[assets_input, liabilities_input],
                    outputs=[net_worth_output, recent_feature_usage] # Outputting the feature details to the state
                )

            # New Tab: Loan Payment Calculator
            with gr.TabItem("🏠 Loan Calculator"):
                gr.Markdown("### Estimate your monthly loan payments")
                with gr.Row():
                    loan_principal = gr.Number(label="Principal Loan Amount ($)", placeholder="200000")
                    annual_rate = gr.Number(label="Annual Interest Rate (%)", placeholder="5.5")
                    loan_years = gr.Number(label="Loan Term (Years)", placeholder="30", precision=0)
                loan_calc_btn = gr.Button("Calculate Monthly Payment 🏠", variant="primary")
                loan_calc_output = gr.Textbox(
                    label="Loan Payment Estimate",
                    lines=5,
                    interactive=False
                )

                loan_calc_btn.click(
                    calculate_loan_payment,
                    inputs=[loan_principal, annual_rate, loan_years],
                    outputs=[loan_calc_output, recent_feature_usage] # Outputting the feature details to the state
                )


            # New Tab: Financial Games
            with gr.TabItem("🎮 Financial Games"):
                gr.Markdown("### Learn finance through interactive games")
                game_selector = gr.Radio(
                    choices=["Budgeting Challenge", "Investment Simulation"],
                    label="Select a Game",
                    value="Budgeting Challenge"
                )
                start_game_btn = gr.Button("Start Game 🎮", variant="secondary")
                game_output = gr.Textbox(label="Game Output", lines=15, interactive=False) # Increased lines for game output


                # Update the click function to capture feature details and update state
                start_game_btn.click(
                    start_selected_game_with_context,
                    inputs=[game_selector, user_profile],
                    outputs=[game_output, recent_feature_usage] # Outputting the feature details to the state
                )


        # Footer
        gr.Markdown("""
        ---
        **Disclaimer**: This AI provides general financial education and guidance. Always consult with qualified financial professionals for major financial decisions.

        **Model**: IBM Granite 3.2-2B Instruct | **Interface**: Gradio | **Platform**: Google Colab
        """)

    return interface

# Launch the application
def main():
    """Main function to launch the application"""
    print("🚀 Starting Financial AI Assistant...")
    print("📱 Model: IBM Granite 3.2-2B Instruct")
    print("🌐 Interface: Gradio")
    print("☁️  Platform: Google Colab")

    # Create and launch interface
    demo = create_interface()

    # Launch with public sharing for Colab
    demo.launch(
        share=True,  # Creates public link for Colab
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()

# Additional utility functions for enhanced functionality - Already included above
# class FinancialCalculators: ...

# Sample usage and testing functions - Already included above
# def test_model(): ...

print("🔧 Financial AI Assistant Setup Complete!")
print("📋 Features included:")
print("  ✅ Personalized Financial Guidance")
print("  ✅ AI-Generated Budget Summaries")
print("  ✅ Spending Insights and Suggestions")
print("  ✅ Demographic-Aware Communication")
print("  ✅ Conversational NLP Experience")
print("  ✅ Spending Visualization")
print("  ✅ Real-time Stock Data Lookup")
print("  ✅ Financial Games (Budgeting Challenge, Investment Simulation)")
print("  ✅ Net Worth Tracker")
print("  ✅ Loan Payment Calculator")

print("\n🚀 Run the main() function to launch the application!")
print("\n💡 Example usage in Colab:")
print("   main()  # This will start the Gradio interface")
