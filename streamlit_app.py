import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

# Page title
st.set_page_config(page_title='Bayesian AB Testing Calculator',
                   page_icon='📊',
                   menu_items={
        'Get Help': 'https://www.linkedin.com/in/daviddiasrodriguez/',
        'Report a bug': "https://www.linkedin.com/in/daviddiasrodriguez/",
        'About': "Made by David Dias Rodríguez. Sterling @ 2025"
    })


st.title('📊 Bayesian AB Testing Calculator')
st.caption('Made by :blue[Sterling]')

st.caption('🚀 :green[New feature:] Upload your CSV files')
st.caption('💪🏾 :blue[Work in progress:] Connect your Google Sheets and BigQuery accounts')

def getAlphaBeta(mu, sigma):
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)

    beta = alpha * (1 / mu - 1)

    return {"alpha": alpha, "beta": beta}

with st.expander('Settings'):
  st.markdown('**What can this app do?**')
  st.caption('This app shows the use of bayesian statistics for AB Testing using the beta distribution. ')

  st.caption('prior: Beta(α, β);  posterior:  Beta(x,y)')
  st.caption('x = NumberOfSuccesses + α')
  st.caption('y = NumberOfObservations - NumberOfSuccesses - β')

  #st.info('This app shows the use of bayesian statistics for AB Testing.')
  st.markdown('**How to use the app?**')
  st.caption('1. Find your optimal beta priors using the Average CR and standard deviation.')
  st.caption('2. Select number of variants you want to analyze.')
  st.caption('3. Add your values and voilà.')
  on = st.toggle("Customize your priors")
  mu_value = 0.045
  sigma_value = 0.1
  if on:
    st.caption("Average CR: 0.045 ; Standard Deviation: 0.1 as pre-values")
    mu_value = st.number_input("Average CR", value=0.045, placeholder="Type the A prior here", min_value=0.0, max_value=50000.0)
    sigma_value = st.number_input("Standard Deviation", value=0.1, placeholder="Type the B prior here", min_value=0.0, max_value=50000.0)


  beta_prior_results = getAlphaBeta(mu_value, sigma_value)
  beta_data = {'alpha': beta_prior_results['alpha'],
 'beta': beta_prior_results['beta']}
  table_beta = pd.DataFrame(data=beta_data, index=[0])
  st.table(table_beta)
  variant_number = st.slider("How many variants you will analyze?", 2, 4, 2)
  type_input_data = st.selectbox('Select your data source', ['Manual','Upload file','Google Sheets / BigQuery Connection'])
  st.divider()  # 👈 Draws a horizontal rule
  
  #manual_input, upload_file, google_connector = st.tabs(["Manual", "Upload file", "Google Sheets /BigQuery Connection"])

  if type_input_data == 'Manual':
    control_users = st.number_input("Control users", value=1000, placeholder="Type the control users here", min_value=0)
    control_purchases = st.number_input("Control interactions", value=50, placeholder="Type the control purchases here", min_value=0)
    v1_users = st.number_input("V1 users", value=1000, placeholder="Type a V1 users here", min_value=0)
    v1_purchases = st.number_input("V1 interactions", value=50, placeholder="Type a V1 purchases here", min_value=0)
    variant_name = ['control', 'v1']
    values_list = [[control_users,control_purchases],
                [v1_users,v1_purchases]]


    if variant_number == 3:
      v2_users = st.number_input("V2 users", value=1000, placeholder="Type a V2 users here", min_value=0)
      v2_purchases = st.number_input("V2 interactions", value=50, placeholder="Type a V2 purchases here", min_value=0)
      variant_name = ['control', 'v1', 'v2']
      values_list = [[control_users,control_purchases],
                  [v1_users,v1_purchases],
                  [v2_users,v2_purchases]]
    if variant_number == 4:
      v2_users = st.number_input("V2 users", value=1000, placeholder="Type a V2 users here", min_value=0)
      v2_purchases = st.number_input("V2 interactions", value=50, placeholder="Type a V2 purchases here", min_value=0)
      v3_users = st.number_input("V3 users", value=1000, placeholder="Type a V3 users here", min_value=0)
      v3_purchases = st.number_input("V3 interactions", value=50, placeholder="Type a V3 purchases here", min_value=0)
      variant_name = ['control', 'v1', 'v2', 'v3']
      values_list = [[control_users,control_purchases],
                  [v1_users,v1_purchases],
                  [v2_users,v2_purchases],
                  [v3_users,v3_purchases]]
      
  if type_input_data == 'Upload file':
    uploaded_file = st.file_uploader("Upload a file (CSV files only*)")
    with st.popover("How to use it", width="stretch"):
      st.markdown('**Upload the data with the following structure**')
      table_example = pd.DataFrame(data={"control_users": 1000, "control_conversions": 50,
                         "v1_users": 1000, "v1_conversions": 50, }, index=[0])
      table_example
      

    pass  # Initialize values_list as an empty list

    if uploaded_file is None:
        pass
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)

        list_descriptions = ['Select the control users', 
                             'Select the control conversions',
                             'Select the v1 users',
                             'Select the v1 conversions',
                             'Select the v2 users',
                             'Select the v2 conversions',
                             'Select the v3 users',
                             'Select the v3 conversions']

        dataframe
        if len(list(dataframe.columns)) >= 4 and variant_number == 2:
            variant_name = ['control', 'v1']
            control_users = int(dataframe[list(dataframe.columns)[0]][0])
            control_purchases = int(dataframe[list(dataframe.columns)[1]][0])
            v1_users = int(dataframe[list(dataframe.columns)[2]][0])
            v1_purchases = int(dataframe[list(dataframe.columns)[3]][0])
            values_list = [[control_users, control_purchases],
                           [v1_users, v1_purchases]]

        elif len(list(dataframe.columns)) >= 6 and variant_number == 3:
            variant_name = ['control', 'v1', 'v2']
            control_users = int(dataframe[list(dataframe.columns)[0]][0])
            control_purchases = int(dataframe[list(dataframe.columns)[1]][0])
            v1_users = int(dataframe[list(dataframe.columns)[2]][0])
            v1_purchases = int(dataframe[list(dataframe.columns)[3]][0])
            v2_users = int(dataframe[list(dataframe.columns)[4]][0])
            v2_purchases = int(dataframe[list(dataframe.columns)[5]][0])
            values_list = [[control_users,control_purchases],
                  [v1_users,v1_purchases],
                  [v2_users,v2_purchases]]

        elif len(list(dataframe.columns)) >= 8 and variant_number == 4:
            variant_name = ['control', 'v1', 'v2', 'v3']
            control_users = int(dataframe[list(dataframe.columns)[0]][0])
            control_purchases = int(dataframe[list(dataframe.columns)[1]][0])
            v1_users = int(dataframe[list(dataframe.columns)[2]][0])
            v1_purchases = int(dataframe[list(dataframe.columns)[3]][0])
            v2_users = int(dataframe[list(dataframe.columns)[4]][0])
            v2_purchases = int(dataframe[list(dataframe.columns)[5]][0])
            v3_users = int(dataframe[list(dataframe.columns)[6]][0])
            v3_purchases = int(dataframe[list(dataframe.columns)[7]][0])
            values_list = [[control_users,control_purchases],
                    [v1_users,v1_purchases],
                    [v2_users,v2_purchases],
                    [v3_users,v3_purchases]]
    
  if type_input_data == 'Google Sheets / BigQuery Connection':
      st.markdown('**@DavidDias is working on it**')
      pass
      
    

   
if 'values_list' in globals():

  beta_simulations = {}

  control_beta_values = np.random.beta(values_list[0][1] + beta_prior_results['alpha'], 
                                  values_list[0][0] - values_list[0][1] + beta_prior_results['beta'], 10000)

  box_plot_results = pd.DataFrame(columns=['variant', 'CR',  'diff'], index=None)

  control_beta_values = np.random.beta(values_list[0][1] + beta_prior_results['alpha'], 
                                  values_list[0][0] - values_list[0][1] + beta_prior_results['beta'], 10000)

  new_df = pd.DataFrame(data={'variant': variant_name[0], 'CR': control_beta_values, 'diff':control_beta_values*0})
  box_plot_results = pd.concat([box_plot_results, new_df], ignore_index=True)
  beta_simulations.update({variant_name[0]: control_beta_values})

  interval_values = {}
  interval_values.update({variant_name[0]: [round(np.percentile(box_plot_results['CR'], 10), 4), round(np.percentile(box_plot_results['CR'], 90), 4)]})

  for i in range(len(values_list[0:])):
      beta_values = np.random.beta(values_list[i][1] + beta_prior_results['alpha'], 
                                  values_list[i][0] - values_list[i][1] + beta_prior_results['beta'], 10000)
      beta_simulations.update({variant_name[i]: beta_values})
      new_df = pd.DataFrame(data={'variant': variant_name[i], 'CR': beta_values, 'diff':beta_values - control_beta_values})
      box_plot_results = pd.concat([box_plot_results, new_df], ignore_index=True)
      interval_values.update({variant_name[i]: [round(np.percentile(new_df['CR'], 10), 4), round(np.percentile(new_df['CR'], 90), 4)]})



  tab0, tab1, tab2 = st.tabs(["All Variants", "Difference vs Control", "Prior and Posterior Data"])



    
  with tab0:
    
    boxplot = px.box(box_plot_results, x="variant", y="CR", color="variant")
    st.plotly_chart(boxplot, use_container_width=True)

  with tab1:
    
    box_plot_results_wo_control = box_plot_results.loc[box_plot_results['variant'] != 'control']
    boxplot_diff = px.box(box_plot_results_wo_control, x="variant", y="diff", color="variant")
    st.plotly_chart(boxplot_diff, use_container_width=True)

  with tab2:
    prior_control_beta_values = np.random.beta(beta_prior_results['alpha'], beta_prior_results['beta'], 10000)
    df_prior_posterior = pd.DataFrame({'prior': prior_control_beta_values,
                        'posterior': control_beta_values})
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=prior_control_beta_values, name='prior'))
    fig.add_trace(go.Histogram(x=control_beta_values, name='posterior'))
    fig.update_layout(barmode='overlay')
      
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig, theme="streamlit")
    table_beta = pd.DataFrame(data=beta_data, index=[0])

  st.subheader('Results:')
  diff_v1 = beta_simulations['v1'] - beta_simulations['control']
  loss_v1 = diff_v1[diff_v1<0]
  loss_list = [((diff_v1 > 0).mean())*100]

  csv = pd.DataFrame(data={'Experiment Name': 'Undefined',
                      'control_users': control_users,
                      'control_conversions': control_purchases,
                        'v1_users': v1_users,
                        'v1_conversions': v1_purchases,
                            'control_percentil_10_cr': interval_values['control'][0],
                            'control_percentil_90_cr': interval_values['control'][1], 
                            'v1_percentil_10_cr': interval_values['v1'][0],
                            'v1_percentil_90_cr': interval_values['v1'][1],
                            'v1_prob_to_win_control': loss_list[0],}, index=[0])


  if variant_number == 3:
    diff_v2 = beta_simulations['v2'] - beta_simulations['control']
    loss_v2 = diff_v2[diff_v2<0]
    loss_list = [((diff_v1 > 0).mean())*100, ((diff_v2 > 0).mean())*100]

    csv = pd.DataFrame(data={'Experiment Name': 'Undefined',
                      'control_users': control_users,
                      'control_conversions': control_purchases,
                        'v1_users': v1_users,
                        'v1_conversions': v1_purchases,
                          'v2_users': v2_users,
                          'v2_conversions': v2_purchases,
                            'control_percentil_10_cr': interval_values['control'][0],
                            'control_percentil_90_cr': interval_values['control'][1], 
                            'v1_percentil_10_cr': interval_values['v1'][0],
                            'v1_percentil_90_cr': interval_values['v1'][1],
                            'v2_percentil_10_cr': interval_values['v2'][0],
                            'v2_percentil_90_cr': interval_values['v2'][1],
                            'v1_prob_to_win_control': loss_list[0],
                            'v2_prob_to_win_control': loss_list[1]}, index=[0])



  if variant_number == 4:
    diff_v2 = beta_simulations['v2'] - beta_simulations['control']
    diff_v3 = beta_simulations['v3'] - beta_simulations['control']
    loss_v2 = diff_v2[diff_v2<0]
    loss_v3 = diff_v3[diff_v3<0]
    loss_list = [((diff_v1 > 0).mean())*100, ((diff_v2 > 0).mean())*100, ((diff_v3 > 0).mean())*100]
    csv = pd.DataFrame(data={'Experiment Name': 'Undefined',
                      'control_users': control_users,
                      'control_conversions': control_purchases,
                        'v1_users': v1_users,
                        'v1_conversions': v1_purchases,
                          'v2_users': v2_users,
                          'v2_conversions': v2_purchases,
                            'v3_users': v3_users,
                          'v3_conversions': v3_purchases,
                            'control_percentil_10_cr': interval_values['control'][0],
                            'control_percentil_90_cr': interval_values['control'][1], 
                            'v1_percentil_10_cr': interval_values['v1'][0],
                            'v1_percentil_90_cr': interval_values['v1'][1],
                            'v2_percentil_10_cr': interval_values['v2'][0],
                            'v2_percentil_90_cr': interval_values['v2'][1],
                            'v3_percentil_10_cr': interval_values['v3'][0],
                            'v3_percentil_90_cr': interval_values['v3'][1],
                            'v1_prob_to_win_control': loss_list[0],
                            'v2_prob_to_win_control': loss_list[1],
                            'v3_prob_to_win_control': loss_list[2]}, index=[0])


  d = {'variant': variant_name[1:],
  'Prob. variant winning Control (%)': loss_list}


  table_bayes_cr = pd.DataFrame(data=d)
  st.table(table_bayes_cr)



  if st.button("Create report"):
    st.download_button(
      "Download Metrics report as CSV", 
      csv.to_csv(index=False).encode("utf-8"), 
      "experiment_report.csv",
      "text/csv",
      key="download-tools-csv",
      icon=":material/download:"
  )
    st.toast('Your report was created!', icon='😍')
  

st.subheader('Guidance about choosing the threshold')

st.caption("The choice of threshold depends on the specific context and the decision-makers' preferences. Here’s a step-by-step approach to determining the threshold:")
st.caption("**Assess the impact:** Consider the potential impact of the change on key business metrics.")
st.caption("**Evaluate costs and reversibility:** Understand the costs involved in implementing the change and how easy it is to revert if needed.")
st.caption("**Gauge risk appetite:** Determine the organization’s tolerance for risk.")
st.caption("**Set the threshold:** Based on the above factors, choose a probability threshold that balances confidence with the need for action.")


st.subheader('Summary')
st.caption("**High threshold (e.g., 95%):** Suitable for high-cost, high-impact decisions with significant consequences.")
st.caption("**Moderate threshold (e.g., 80%):** Suitable for balanced decision-making where there’s a moderate impact.")
st.caption("**Low threshold (e.g., 70%):** Suitable for low-cost, easily reversible decisions with minor impact.")

st.caption("Do you wanna contribute to this project?")
st.link_button("Donate", "https://donate.stripe.com/6oEaFe4TeegDcF25kl")

st.caption("Do you need an experimentation program?")
st.link_button("Contact us", "https://sterlingdata.webflow.io/company/contact?tool=bayesian_ab_testing_calculator")


st.caption('Sterling @ 2025')
st.caption('Updated: 02/07/2025')