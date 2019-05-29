using Base.Iterators: partition

function img_load(img_name)
    img = load(img_name)
    img = Float64.(channelview(img))
    return img
end

function load_dataset_as_batches(path, BATCH_SIZE)
    data = []
    for r in readdir(path)
        img_path = string(path, r)
        push!(data, img_load(img_path))
    end
    num_images = length(data)
    return [reshape(hcat(data...), 64, 64, 3, num_images) for imgs in partition(imgs, BATCH_SIZE)]
end 
