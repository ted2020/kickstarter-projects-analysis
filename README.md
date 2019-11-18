### kickstarter projects analysis
 ######## explore the data
 ######## visualize the data
 ######## wordcloud and wordcloud prediction
 ######## logistic regression
 ######## randomforest


#### Kickstarter Projects Analysis
Ted Dogan

Abstract

Crowdsourcing platforms allows individuals with creative ideas to reach out to vast number of potential investors, including all scales of contributions. The dataset that’s been used for this project is from Kaggle.com. It includes near 400 thousands projects and 25 variables for each project. Aim of this is to create a prediction algorithm whether a project will be successful amongst the diverse universe of projects. The analysis gives an idea about the possible high funding times of a year and a month, countries that are using the platform the most and currencies that are the most preferred, what sorts of projects receive excess funding by main and sub-category, time lengths of projects, and wordcloud of projects’ titles. Finally, I predict the successful projects by using logistic regression, randomforest, and wordcloud keywords. The results provide a simplistic understanding of the platform and the ways the creators may modify their ideas to be funded.

Executive Summary

Being able to reach out to clients or investors is one of the many great advantages of technology. Kickstarter enables makers to put forward their ideas and hopefully attain the attention of investors, who are on the lookout for the next big thing. Easiness of the platform, along with its early entrant market advantage, make Kickstarter a unique environment. There are many projects here, relatable to almost every aspect of life. But the changing preferences of people, with their desire of differentiated products can be observed in the data. Also, some sub-categories and categories are the leaders in terms of backers and percentage of fundedness. I will try to break it down as much as possible.
Kickstarter is a crowdsourcing platform to encourage creative projects for those who don't have much financial means. It's one of the go-to places for venture capitalists and angel investors. Even individuals can take part in projects with small capitals. The platform includes variety of projects from biotech to painting. If one puts forward a project for funding, that person is called a "creator." Each creator must go through an immense task of providing a thorough analysis and objectives of the project. Outline of a project should indicate the stage, steps to be taken, possible outcome(s), use of it, and provide all the links and files, so that the investors can make an informed decision. If an investor decides to pledge an amount, he/she is called a "backer" and the amount contributed is named as "pledged amount." If the total asked funding by the creator has not been achieved, the pledged amounts of investors are not collected, and the project doesn't go through. So, it's an all or none model. Therefore, creators should put strong creativity, research, and sincerity into their projects before they decide to go on to the platform of Kickstarter.
This analysis mostly focuses on the excess funding, rather than just fundedness. Therefore, the main attention is on the how much more a project is funded over its asked pledged amount. Most music, dance, theatre, art, and design categories and their sub-categories receive most than the initial set amount. The results exclude some assumptions, such as: the quality of a project, where the project is initiated in, creators’ social status and social media presence, potential fraud status, motivations of backers and creators, factors of over funding, and the reasonableness of the asked pledged amount.

Literature

Crowdfunding platforms are one of the biggest phenomena of the last decade. Kickstarter was launched in 2009 and since then, thousands of projects by many creators have been up. Yet still, there are not many viable scholarly studies that’s been written on this subject, though there are many thorough analyses that’s been done by the analysts. Two of a few studies that I find useful for my case are: “Crowdfunding: Tapping the right crowd” by Belleflamme and Lambert, and “Crowdfunding Creative Ideas: The Dynamics of Project Backers in Kickstarter” by Kuppuswamy and Bayus. First study makes a comparison of crowdfunding projects based on pre-order and equity issue, and ways of creating a substantial backers community. Second study focuses on backers and their preferences over a time period. These papers provide an intuition into the crowdfunding platforms, yet there is a need for detailed research both from the perspective of backers and creators.

Data and Methods

I used the data on Kickstarter that’s been provided on Kaggle.com. The analysis consists of four parts: exploratory, visual, wordcloud, and predicting. Before working with data, I checked the missing values. Only some of the usd.pledge column’s data are missing. Since I am not going to use this variable in my analysis, I didn’t see a need for correcting it. For the exploratory part, other than just seeing the levels, I created new variables from the timestamp column, in addition to calculating the excess fund of each project. Then, I found the total days that the project has been on the platform, open for funding. This process has been applied to the main and sub-categories. And lastly, I observed the countries that are using the platform the most and the most preferable currencies in transacting the business. 
 
 

For the visual part, by using the R’s ggplot functionality, I created excess funding by the main and sub-category, by the month and weekday. These visuals help the audience to better grasp the intuition of the dataset. Most over funded categories include music, art, dance, design, and theater, and their sub-categories. There is also a bump in funding on Mondays and Fridays for the reasons that I do not know yet.
  
  
For the wordcloud part, by using the titles of projects, I extracted the most common words that are used to see whether a pattern does exist. Since from the visual part, I know which categories are the leaders of excess funding, therefore I assumed that the extracted title words should reflect the similar sentiment and they do. Most common words are “new”, “album”, “film”, “book”, “game”, “art”, and “music”. 
  
For the predicting part, I try to predict whether a project will be successful. To accomplish this, first, I used logistic regression, second, randomforest regression, and lastly wordcloud bing lexicon sentiment.
 
 
 

Results

There are many limitations to this analysis, as mentioned in summary section. But, due to data limitation and not to replicate the others’ work, I am applying the excess funding technique. 
I, first, explored the data and created variables that I thought could be useful. Then, I created some visualizations to better grasp the intuition of the dataset. In addition, I created a wordcloud for the project titles whether there is a common theme. And lastly, I tried to predict the success/fail likelihood of a project by using the logistic regression and randomforest regression.
I used logistic model by using the variables that I believe to be correcting whether the project will go through. First, I converted successful attempts in the Kickstarter dataset to 1, and all the others to 0. Then, I set my equation with the independent variables that I found, in this case, to better predict. These variables do not carry multicollinearity and all statistically significant. Therefore, I move on with them. Of course, this is a preliminary result but to give an idea of what can be done, this is a good approach. Although the model I created is far from perfect, I can predict whether the project will be successful at 84% of the time by using logistic regression.
I also tried randomforest, which improves the prediction accuracy just a bit. It's stable at 85.4%.
This prediction can be extended predicting a project from wordcloud sentiment analysis. By working with only the titles, no description, 30% accuracy can be obtained. I assume that including a detailed description of each project and breaking it down to wordcloud analysis, the accuracy can be improved substantially. 

Conclusion

Crowdfunding platforms provide an essential connection between the creators and backers, who are in many cases also referred as angel investors and venture capitalists. Online funding platforms are mostly overlooked and not studied thoroughly. They, in a way, altered the way of funding. Aim of this analysis is to give an idea about what Kickstarter is, what are the excess funded categories, any interesting times of the year to look for a funding, and providing a tool for predicting the end result.  This is not to give an advice, but more to show what can be accomplished. Limitations to overcome are stated in the summary section. Many other analysts didn’t pay much attention to excess funding, that’s the reason why I went with it, to see the issue from a different perspective. Further exploratory analysis, along with graphical results, are required to provide a more meaningful approach into what makes a project successful. If limitations can be substantially reduced, there is a chance of creating a resume-like crowdfunding template for creators. That, in turn, might help investors in their quest to find the next big thing, speed-up the process by searching through the keywords in those templates, just as human resource departments do.






References

Bellaflamme and Lambert. “Crowdfunding: Tapping the Right Crowd.” Journal of Business Venturing, Elsevier, 29 Sept. 2013, www.sciencedirect.com/science/article/pii/S0883902613000694.

Kuppuswamy and Bayus. “Crowdfunding Creative Ideas: The Dynamics of Project Backers in Kickstarter.” SSRN, 17 Mar. 2013, papers.ssrn.com/sol3/papers.cfm?abstract_id=2234765.

Mouillé, Mickaël. “Kickstarter Projects.” Kaggle, 8 Feb. 2018, www.kaggle.com/kemical/kickstarter-projects/version/7.

Saha, Srishti. “Will Your Kickstarter Project Be Successful? A Simple Analysis to Help You Predict Better!!!” Good Audience, 14 Sept. 2018, blog.goodaudience.com/kickstarter-projects-prediction-of-state-steps-for-a-beginner-analysis-f4630a50b7fe.

