using Base.Iterators: partition
using Images

function img_load(img_name)
    img = load(img_name)
    img = imresize(img, 64, 64)
    img = reshape(Float64.(channelview(img)), 64, 64, 3)
    return img
end

function load_dataset_as_batches(path, BATCH_SIZE)
    data = []
    for r in readdir(path)
        img_path = string(path, r)
        push!(data, img_load(img_path))
    end
    num_images = length(data)
    #println(num_images)
    batched_data = []
    for x in partition(data, BATCH_SIZE)
        x = reshape(cat(x..., dims = 4), 64, 64, 3, BATCH_SIZE)
        push!(batched_data, x)
    end
    return batched_data
end
