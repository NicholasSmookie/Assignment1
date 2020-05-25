using CSV, DataFrames, Plots

avi = CSV.read("C:/Users/Smookie/Documents/My NUST/2020/AIG/Assignment/bank-additional-full.csv")
avi = convert(Matrix,avi)

size(avi)[2]
for i = 1:(size(avi)[2]-2)
    sort_order = unique(avi[:,i])
    for j = 1:size(avi)[1]
        for k = 1:size(sort_order)[1]
            if avi[j,i] == sort_order[k]
                avi[j,i] = Int(k)
            end
        end
    end
    println(i)
end

convert_set = avi[:,end-2:end]
set = fill(1, size(avi)[1])
for i = 1:size(convert_set)[1]
    if sum(convert_set[i,:])/3.0 >= 10
        set[i] = 1
    else set[i] = 0
    end
end
test_data = avi[:,end-2]+avi[:,end-1]+avi[:,end]
test_data./3

concluded = avi[:,1:end-2]
values = set

indd = ceil(0.8*size(concluded,1))
splitt = convert(Int,indd)

train_data = concluded[1:splitt,:]
train_values = values[1:splitt]
bias = fill(1, splitt)
train_data = hcat(train_data,bias)
train_data = train_data./maximum(train_data)

function sigmoid(z)
    return 1 ./ (1 .+ exp.(.-z))
end

function regularised_cost(X, y, θ, λ)
    y = size(train_data,1)
    m = length(y)
    h = sigmoid(X * θ)

    positive_cost = ((-y)' * log.(h))
    negative_cost = ((1 .- y)' * log.(1 .- h))

    lambda_regularization = (λ/(2*m) * sum(θ[2 : end] .^ 2))

    batch_cost = (1/m) * (positive_cost - negative_cost) + lambda_regularization

    gradient = (1/m) * (X') * (h-y) + ((1/m) * (λ * θ))

    gradient[1] = (1/m) * (X[:, 1])' * (h-y)

    return (batch_coast, gradient)
end
