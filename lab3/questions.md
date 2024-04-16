### Questions and discussion

## What I found easy to do in Dash:

I found it easy to map the layout component IDs to the callback function inputs and outputs. This was very intuitive and easy to understand. As I was already familiar with web development, I found it easy to understand the layout and the components of the Dash app.

API integration was also seamless and easy to implement using the Request library.

## What was hard to implement or you didn’t wind up getting it to work?

Understanding the callback function and how it works was not that easy for a new user. It took me some time to understand how to map the input and output components. However, once I understood it, it was easy to implement.

The most difficult thing to grasp was how to let the UI reflect the changes triggered by the callback function. I had to spend some time to understand how to update the UI based on the callback function output. Maneuvering the UI based on the callback function output was a bit tricky. But, I got an idea of using the hidden div to store the data and then use it to update the UI.

## What other components, or what revised dashboard design, would you suggest to better assist in explaining the behavior of the Iris model to a client?

The dashboard looks quite good but it can be improved by:

- Showing available datasets description where the user can just select which dataset they want to use. This can avoid the user from uploading the dataset again and again. Also, the user can see the description of the dataset and then decide which dataset they want to use.

- Showing available models and their description can also be helpful. The user can select the model they want to use and then see the model description. This can help the user to understand the model and its behavior.

- Showing the progress bar while the model is training can also be helpful. The user can see the progress of the model training and can understand how much time it will take to train the model.

- Allowing the user to choose different hyperparameters can also be helpful. The user can select the hyperparameters they want to use and then see the model performance based on the hyperparameters. This can help the user to understand how the model performance changes based on the hyperparameters.

- On the visualization part, if the user could choose which kind of plots or visualization they want to see, it can increase user interaction and user experience. The user can select the visualization they want to see and then see the visualization based on their selection. This can help the user to understand the data and model better.

## Can you think of better ways to link the “back end” Iris model and its results with the front-end Dash functions?
I think the current implementation is good but it can be improved by:
- modifying design perspective to make it more user-friendly.
- Using database or other storage (Persistence) to store the model and its results. This can help the user to don't train the model again and again. The user can just select the model and see the results. This can save time and resources.
- Using the cache to store the model and its results. This can help the user to see the results faster as the model and its results are already stored in the cache. This can save time and resources.

## COnclusion
Overall, I found Dash to be a very powerful tool for building interactive web applications. It is easy to use and understand. The documentation is also very good and easy to follow. I would recommend Dash to anyone who wants to build interactive dashboards or statstical web applications. But when it comes to building a complex web application, it can be a bit tricky and satsifying the user requirements can be a bit challenging.