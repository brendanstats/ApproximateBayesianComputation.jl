"""
Converts a Base.Dates.XX format into a string in HH:MM:SS format.
"""
function duration_to_string(duration::Base.Dates.Hour)
    duration = convert(Int64, duration)
    return string(duration)
end

function duration_to_string(duration::Base.Dates.Minute)
    duration = convert(Int64, duration)
    minutes = duration % 60
    hours = (duration - minutes) / 60
    minutes = convert(Int64, minutes)
    hours = convert(Int64, hours)
    return string(hours, ":", minutes)
end

function duration_to_string(duration::Base.Dates.Second)
    duration = convert(Int64, duration)
    seconds = duration % 60
    duration = (duration - seconds) / 60
    minutes = duration % 60
    hours = (duration - minutes) / 60
    seconds = convert(Int64, seconds)
    minutes = convert(Int64, minutes)
    hours = convert(Int64, hours)
    return string(hours, ":", minutes, ":", seconds)
end

function duration_to_string(duration::Base.Dates.Millisecond)
    duration = convert(Int64, duration)
    duration = floor(duration / 1000)
    seconds = duration % 60
    duration = (duration - seconds) / 60
    minutes = duration % 60
    hours = (duration - minutes) / 60
    seconds = convert(Int64, seconds)
    minutes = convert(Int64, minutes)
    hours = convert(Int64, hours)
    return string(hours, ":", minutes, ":", seconds)
end

"""
Returns only the rows (2D) or entries corresponding to the supplied indicies
"""
function subset{T <: Any, G <: Integer}(A::Array{T, 1}, idxs::Array{G})
    return A[idxs]
end

function subset!{T <: Any, G <: Integer}(A::Array{T, 1}, idxs::Array{G})
    A = A[idxs]
    nothing
end

function subset{T <: Any, G <: Integer}(A::Array{T, 2}, idxs::Array{G})
    return A[idxs, :]
end

function subset!{T <: Any, G <: Integer}(A::Array{T, 2}, idxs::Array{G})
    A = A[idxs, :]
    nothing
end
