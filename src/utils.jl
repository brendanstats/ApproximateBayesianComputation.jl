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

#=
"""
Returns the inverse of a permutaitons `x[perm][invperm(perm)]` will be `x`
"""
function invperm{G <: Real}(perm::Array{G, 1})
    out = Array{G}(length(perm))
    for (ii, x) in enumerate(perm)
        out[x] = ii
    end
    return out
end
=#
#invpermx[permy[4]]
