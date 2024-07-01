import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
#boxplot.show()

#st.plotly_chart(fig, use_container_width=True)

# Page title
st.set_page_config(page_title='Bayesian AB Testing Calculator', page_icon='📊')
st.title('📊 Bayesian AB Testing Calculator')

def getAlphaBeta(mu, sigma):
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)

    beta = alpha * (1 / mu - 1)

    return {"alpha": alpha, "beta": beta}

with st.expander('Settings'):
  st.markdown('**What can this app do?**')
  st.caption('This app shows the use of bayesian statistics for AB Testing using the beta distribution.')
  st.latex(r'''
    x = NumberOfSuccesses + a
    ''')
  st.latex(r'''
    y = NumberOfObservations - NumberOfSuccesses - b
    ''')
  #st.info('This app shows the use of bayesian statistics for AB Testing.')
  st.markdown('**How to use the app?**')
  st.caption('1. Find your optimal beta priors')
  st.caption('2. Select number of variants you want to analyze.')
  st.caption('3. Add your values and voilà.')
  mu_value = st.number_input("Average CR", value=0.045, placeholder="Type the A prior here", min_value=0.0, max_value=50000.0)
  sigma_value = st.number_input("Standard Deviation", value=0.1, placeholder="Type the B prior here", min_value=0.0, max_value=50000.0)
  beta_prior_results = getAlphaBeta(mu_value, sigma_value)
  #st.warning('To engage with the app: <br />1. Find your optimal beta priors <br /> 2. Select number of variants you want to analyze. <br /> 3. Add your values and voilá.')
  variant_number = st.slider("How many variants you will analyze?", 2, 4, 2)
  control_users = st.number_input("Control users", value=1000, placeholder="Type the control users here", min_value=0)
  control_purchases = st.number_input("Control purchases", value=50, placeholder="Type the control purchases here", min_value=0)
  v1_users = st.number_input("V1 users", value=1000, placeholder="Type a V1 users here", min_value=0)
  v1_purchases = st.number_input("V1 purchases", value=50, placeholder="Type a V1 purchases here", min_value=0)
  variant_name = ['control', 'v1']
  values_list = [[control_users,control_purchases],
               [v1_users,v1_purchases]]

  if variant_number == 3:
   v2_users = st.number_input("V2 users", value=1000, placeholder="Type a V2 users here", min_value=0)
   v2_purchases = st.number_input("V2 purchases", value=50, placeholder="Type a V2 purchases here", min_value=0)
   variant_name = ['control', 'v1', 'v2']
   values_list = [[control_users,control_purchases],
               [v1_users,v1_purchases],
               [v2_users,v2_purchases]]
  if variant_number == 4:
   v2_users = st.number_input("V2 users", value=1000, placeholder="Type a V2 users here", min_value=0)
   v2_purchases = st.number_input("V2 purchases", value=50, placeholder="Type a V2 purchases here", min_value=0)
   v3_users = st.number_input("V3 users", value=1000, placeholder="Type a V3 users here", min_value=0)
   v3_purchases = st.number_input("V3 purchases", value=50, placeholder="Type a V3 purchases here", min_value=0)
   variant_name = ['control', 'v1', 'v2', 'v3']
   values_list = [[control_users,control_purchases],
               [v1_users,v1_purchases],
               [v2_users,v2_purchases],
               [v3_users,v3_purchases]]





#control_users = st.number_input("Control users", value=1000, placeholder="Type the control users here", min_value=0)
#control_purchases = st.number_input("Control purchases", value=50, placeholder="Type the control purchases here", min_value=0)
#v1_users = st.number_input("V1 users", value=1000, placeholder="Type a V1 users here", min_value=0)
#v1_purchases = st.number_input("V1 purchases", value=50, placeholder="Type a V1 purchases here", min_value=0)
#variant_name = ['control', 'v1']
#values_list = [[control_users,control_purchases],
#               [v1_users,v1_purchases]]

#if variant_number == 3:
#   v2_users = st.number_input("V2 users", value=1000, placeholder="Type a V2 users here", min_value=0)
#   v2_purchases = st.number_input("V2 purchases", value=50, placeholder="Type a V2 purchases here", min_value=0)
#   variant_name = ['control', 'v1', 'v2']
#   values_list = [[control_users,control_purchases],
#               [v1_users,v1_purchases],
#               [v2_users,v2_purchases]]
#if variant_number == 4:
#   v2_users = st.number_input("V2 users", value=1000, placeholder="Type a V2 users here", min_value=0)
#   v2_purchases = st.number_input("V2 purchases", value=50, placeholder="Type a V2 purchases here", min_value=0)
#   v3_users = st.number_input("V3 users", value=1000, placeholder="Type a V3 users here", min_value=0)
#   v3_purchases = st.number_input("V3 purchases", value=50, placeholder="Type a V3 purchases here", min_value=0)
#   variant_name = ['control', 'v1', 'v2', 'v3']
#   values_list = [[control_users,control_purchases],
#               [v1_users,v1_purchases],
#               [v2_users,v2_purchases],
#               [v3_users,v3_purchases]]


beta_simulations = {}
#beta_prior_a = 10
#beta_prior_b = 30

control_beta_values = np.random.beta(values_list[0][1] + beta_prior_results['alpha'], 
                                 values_list[0][0] - values_list[0][1] + beta_prior_results['beta'], 10000)

box_plot_results = pd.DataFrame(columns=['variant', 'CR',  'diff'], index=None)

control_beta_values = np.random.beta(values_list[0][1] + beta_prior_results['alpha'], 
                                 values_list[0][0] - values_list[0][1] + beta_prior_results['beta'], 10000)

new_df = pd.DataFrame(data={'variant': variant_name[0], 'CR': control_beta_values, 'diff':control_beta_values*0})
box_plot_results = pd.concat([box_plot_results, new_df], ignore_index=True)
beta_simulations.update({variant_name[0]: control_beta_values})

for i in range(len(values_list[0:])):
    
    beta_values = np.random.beta(values_list[i][1] + beta_prior_results['alpha'], 
                                 values_list[i][0] - values_list[i][1] + beta_prior_results['beta'], 10000)
    beta_simulations.update({variant_name[i]: beta_values})
    new_df = pd.DataFrame(data={'variant': variant_name[i], 'CR': beta_values, 'diff':beta_values - control_beta_values})
    box_plot_results = pd.concat([box_plot_results, new_df], ignore_index=True)


tab0, tab1, tab2 = st.tabs(["Prior and Posterior Data", "All Data", "Difference vs Control"])

with tab0:
  prior_control_beta_values = np.random.beta(beta_prior_results['alpha'], beta_prior_results['beta'], 10000)
  df_prior_posterior = pd.DataFrame({'prior': prior_control_beta_values,
                       'posterior': control_beta_values})
  fig = go.Figure()
  fig.add_trace(go.Histogram(x=prior_control_beta_values, name='prior'))
  fig.add_trace(go.Histogram(x=control_beta_values, name='posterior'))
  fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
  fig.update_traces(opacity=0.75)
  st.plotly_chart(fig, theme="streamlit")
  
with tab1:
  st.subheader('Boxplot graph:')
  boxplot = px.box(box_plot_results, x="variant", y="CR", color="variant")
  st.plotly_chart(boxplot, use_container_width=True)

with tab2:
  st.subheader('Boxplot graph:')
  box_plot_results_wo_control = box_plot_results.loc[box_plot_results['variant'] != 'control']
  boxplot_diff = px.box(box_plot_results_wo_control, x="variant", y="diff", color="variant")
  st.plotly_chart(boxplot_diff, use_container_width=True)



#st.subheader('Boxplot graph:')
#boxplot = px.box(box_plot_results, x="variant", y="CR", color="variant")
#st.plotly_chart(boxplot, use_container_width=True)
st.subheader('Results:')

diff_v1 = beta_simulations['v1'] - beta_simulations['control']
loss_v1 = diff_v1[diff_v1<0]
loss_list = [((diff_v1 > 0).mean())*100]

if variant_number == 3:
  diff_v2 = beta_simulations['v2'] - beta_simulations['control']
  loss_v2 = diff_v2[diff_v2<0]
  loss_list = [((diff_v1 > 0).mean())*100, ((diff_v2 > 0).mean())*100]


if variant_number == 4:
   diff_v2 = beta_simulations['v2'] - beta_simulations['control']
   diff_v3 = beta_simulations['v3'] - beta_simulations['control']
   loss_v2 = diff_v2[diff_v2<0]
   loss_v3 = diff_v3[diff_v3<0]
   loss_list = [((diff_v1 > 0).mean())*100, ((diff_v2 > 0).mean())*100, ((diff_v3 > 0).mean())*100]

d = {'variant': variant_name[1:],
 'Prob. variant winning Control (%)': loss_list}


table_bayes_cr = pd.DataFrame(data=d)
st.table(table_bayes_cr)

st.subheader('Guidance about choosing the threshold')

st.caption("The choice of threshold depends on the specific context and the decision-makers' preferences. Here’s a step-by-step approach to determining the threshold:")
st.caption("**Assess the Impact:** Consider the potential impact of the change on key business metrics.")
st.caption("**Evaluate Costs and Reversibility:** Understand the costs involved in implementing the change and how easy it is to revert if needed.")
st.caption("**Gauge Risk Appetite:** Determine the organization’s tolerance for risk.")
st.caption("**Set the Threshold:** Based on the above factors, choose a probability threshold that balances confidence with the need for action.")


st.subheader('Summary')
st.caption("**High threshold (e.g., 95%):** Suitable for high-cost, high-impact decisions with significant consequences.")
st.caption("**Moderate threshold (e.g., 80%):** Suitable for balanced decision-making where there’s a moderate impact.")
st.caption("**Low threshold (e.g., 70%):** Suitable for low-cost, easily reversible decisions with minor impact.")

st.caption("Do you wanna contribute to this project?")
st.link_button("Donate", "https://donate.stripe.com/5kAbLQ0Nk1v85nqfZ0")